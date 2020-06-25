# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE training (RGCN encoder, GRU decoder, teacher forced decoding). 

To resume training form a given 
- iteration saved
- learning rate
- beta 
pass corresponding args + load_model = True

***
Training script to train on large data : iterate on csv chunks until convergence

"""

import argparse
import sys, os
import torch
import numpy as np

import pandas as pd
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(script_dir)

from utils import ModelDumper, disable_rdkit_logging, setup, log_reconstruction
from dgl_utils import send_graph_to_device
from model_zinc import Model
from loss_func import VAELoss, weightedPropsLoss, affsRegLoss, affsClassifLoss
from dataloaders.molDataset import molDataset, Loader

from time import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='zinc')  # model name in results/saved_models/
    parser.add_argument('--train', help="path to training dataframe", type=str, default='data/shuffled_whole_zinc.csv')
    parser.add_argument("--chunk_size", help="Nbr of molecules loaded simultaneously in memory (csv chunk)", type=int,
                        default=10000)

    # Alphabets params 
    parser.add_argument('--decode', type=str, default='selfies')  # language used : 'smiles' or 'selfies'
    parser.add_argument('--alphabet_name', type=str,
                        default='zinc_alphabets.json')  # name of alphabets json file, in map_files dir
    parser.add_argument('--build_alphabet', action='store_true')  # Ignore json alphabet and build from scratch

    # If we start from a pretrained model :
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_name', type=str, default='default')  # name of model to load from
    parser.add_argument('--load_iter', type=int, default=0)  # resume training at optimize step n°

    # Model architecture 
    parser.add_argument('--n_gcn_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
    parser.add_argument('--n_gru_layers', type=int, default=4)  # number of gcn encoder layers (3 or 4?)
    parser.add_argument('--gcn_dropout', type=float, default=0.2)
    parser.add_argument('--gcn_hdim', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=56) # jtvae uses 56
    parser.add_argument('--gru_hdim', type=int, default=450)
    parser.add_argument('--gru_dropout', type=float, default=0.2)
    parser.add_argument('--use_batchNorm', type=bool, default=True)

    # Training schedule params :
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)  # Initial learning rate
    parser.add_argument('--anneal_rate', type=float, default=0.9)  # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=40000)  # update learning rate every _ step
    
    parser.add_argument('--clip_norm', type=float, default=50.0)  # Gradient clipping max norm
    # Kl weight schedule 
    parser.add_argument('--beta', type=float, default=0.0)  # initial KL annealing weight
    parser.add_argument('--step_beta', type=float, default=0.002)  # beta increase per step
    parser.add_argument('--max_beta', type=float, default=0.5)  # maximum KL annealing weight
    parser.add_argument('--warmup', type=int, default=40000)  # number of steps with only reconstruction loss (beta=0)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)  # update beta every _ step


    # Logging :
    parser.add_argument('--print_iter', type=int, default=1000)  # print loss metrics every _ step
    parser.add_argument('--save_iter', type=int, default=1000)  # save model weights every _ step

    # teacher forcing rnn schedule
    parser.add_argument('--tf_init', type=float, default=1.0) # teacher forcing fraction at the start of training 
    parser.add_argument('--tf_step', type=float, default=0.002)  # step decrease
    parser.add_argument('--tf_end', type=float, default=1.0)  # final tf frequency
    parser.add_argument('--tf_anneal_iter', type=int, default=1000)  # nbr of iters between each annealing
    parser.add_argument('--tf_warmup', type=int, default=70000)  # nbr of steps before tf starts to decrease 

    # Multitask :
    parser.add_argument('--no_props', action='store_false')  # No multitask props
    
    # Other : 
    parser.add_argument('--gpu_id', type=int, default=0)  # run model on cuda:{id} if multiple gpus on server
    parser.add_argument('--processes', type=int, default=20)  # num workers

    # =======

    args, _ = parser.parse_known_args()
    
    if args.n_gru_layers ==4 :
        from model_zinc_4layers import Model

    logdir, modeldir = setup(args.name, permissive=True)
    dumper = ModelDumper(dumping_path=os.path.join(modeldir, 'params.json'), argparse=args)

    use_props, use_affs = True, False
    if args.no_props:
        use_props = False
    
    # Multitasking : properties and affinities should be in input dataset
    if use_props:
        properties = ['QED', 'logP', 'molWt']
    else:
        properties = []
    props_weights = [1e3, 1e2, 1]


    targets = []
    

    writer = SummaryWriter(logdir)
    disable_rdkit_logging()  # function from utils to disable rdkit logs

    # Empty loader object 
    loaders = Loader(maps_path='map_files/',
                     csv_path=None,
                     vocab=args.decode,
                     build_alphabet=args.build_alphabet,
                     alphabet_name=args.alphabet_name,
                     n_mols=-1,
                     num_workers=args.processes,
                     batch_size=args.batch_size,
                     props=properties,
                     targets=targets,
                     redo_selfies = False) # recompute selfies in the dataloader instead of using dataframe value 

    # Model & hparams
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu' # multiple GPU in argparse 

    params = {'features_dim': loaders.dataset.emb_size,  # node embedding dimension
              'num_rels': loaders.num_edge_types,
              'gcn_layers': args.n_gcn_layers,
              'gcn_dropout':args.gcn_dropout,
              'gcn_hdim': args.gcn_hdim,
              'gru_hdim':args.gru_hdim,
              'gru_dropout': args.gru_dropout,
              'batchNorm': args.use_batchNorm, 
              'l_size': args.latent_size,
              'voc_size': loaders.dataset.n_chars,
              'max_len': loaders.dataset.max_len,
              'N_properties': len(properties),
              'N_targets': len(targets),
              'device': device,
              'index_to_char': loaders.dataset.index_to_char,
              'props': properties,
              'targets': targets}
    # pickle.dump(params, open('saved_models/model_params.pickle', 'wb'))
    dumper.dic.update(params)
    dumper.dump()

    model = Model(**params).to(device)

    load_model = args.load_model
    load_path = f'results/saved_models/{args.load_name}/params.json'
    if load_model:
        print(f"Careful, I'm loading {args.load_name} in train.py, line 160")
        weights_path = f'results/saved_models/{args.load_name}/weights.pth'
        model.load_state_dict(torch.load(weights_path))

    print(model)
    map = ('cpu' if device == 'cpu' else None)

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print("> learning rate: %.6f" % scheduler.get_lr()[0])

    # Train & test

    if args.load_model:
        total_steps = args.load_iter
        beta = min(args.beta + args.step_beta*(total_steps-args.warmup)/args.kl_anneal_iter, args.max_beta) # resume at coherent beta, cap at max_beta. 
        print(f'Training resuming at step {total_steps}, beta = {beta}')
    else:
        total_steps = 0
        beta = args.beta
        
    tf_proba = args.tf_init
    start = time() # get training start time 

    for epoch, chunk in enumerate(pd.read_csv(os.path.join(script_dir, args.train), chunksize=args.chunk_size)):

        # give csv chunk to loader 
        loaders.dataset.pass_dataset(chunk, graph_only=False)
        train_loader, _, test_loader = loaders.get_data()

        model.train()
        epoch_train_rec, epoch_train_kl, epoch_train_pmse, epoch_train_amse = 0, 0, 0, 0

        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(train_loader):

            total_steps += 1  # count training steps

            smiles = smiles.to(device)
            graph = send_graph_to_device(graph, device)
            if use_props:
                p_target = p_target.to(device).view(-1, len(properties))

            # Forward passs
            mu, logv, _, out_smi, out_p = model(graph, smiles, tf=tf_proba)

            # Compute loss terms : change according to multitask setting
            rec, kl = VAELoss(out_smi, smiles, mu, logv)

            if not use_props:  # VAE only
                pmse = torch.tensor(0)
            elif use_props :
                pmse = weightedPropsLoss(p_target, out_p, props_weights)

            # COMPOSE TOTAL LOSS TO BACKWARD
            if total_steps < args.warmup:  # Only reconstruction (warmup)
                t_loss = rec
            else:
                t_loss = rec + beta * kl + pmse 

            optimizer.zero_grad()
            t_loss.backward()
            del t_loss
            clip.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # Annealing KL and LR
            if total_steps % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_steps % args.kl_anneal_iter == 0 and total_steps >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)

            if total_steps % args.tf_anneal_iter == 0 and total_steps >= args.tf_warmup:
                tf_proba = max(args.tf_end, tf_proba - args.tf_step)  # tf decreases until reaches tf_end
                print('Updated tf rate: ', tf_proba)
                writer.add_scalar('tf_proba/train', tf_proba, total_steps)

            # logs and monitoring
            if total_steps % args.print_iter == 0:
                now = time()
                print(
                    f'Opt step {total_steps}, rec: {rec.item():.2f}, kl: {beta * kl.item():.2f}, props mse: {pmse.item():.2f}')
                print('Time elapsed : ', now-start)
                
                writer.add_scalar('BatchRec/train', rec.item(), total_steps)
                writer.add_scalar('BatchKL/train', kl.item(), total_steps)
                if use_props:
                    writer.add_scalar('BatchPropMse/train', pmse.item(), total_steps)

            if total_steps % args.save_iter == 0:
                model.cpu()
                torch.save(model.state_dict(), os.path.join(modeldir, "weights.pth"))
                model.to(device)
                
            # Quality of reconstruction in last batch of epoch
            if batch_idx == len(train_loader)-1:
                _, out_chars = torch.max(out_smi.detach(), dim=1)
                """
                # Get 3 input -> output prints for visual check
                _, frac_valid = log_reconstruction(smiles, out_smi.detach(),
                                                   loaders.dataset.index_to_char,
                                                   string_type=args.decode)
                print(f'{frac_valid} valid smiles in batch')
                """
                # Correctly reconstructed characters
                differences = 1. - torch.abs(out_chars - smiles)
                differences = torch.clamp(differences, min=0., max=1.).double()
                quality = 100. * torch.mean(differences)
                quality = quality.detach().cpu()
                writer.add_scalar('quality/train', quality.item(), total_steps)
                print('fraction of correct characters at reconstruction // train : ', quality.item())

            # keep track of epoch loss
            epoch_train_rec += rec.item()
            epoch_train_kl += kl.item()
            epoch_train_pmse += pmse.item()

        # Validation pass
        model.eval()
        val_rec, val_kl, val_amse, val_pmse = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):

                smiles = smiles.to(device)
                graph = send_graph_to_device(graph, device)

                if use_props:
                    p_target = p_target.to(device).view(-1, len(properties))

                mu, logv, z, out_smi, out_p= model(graph, smiles, tf=tf_proba)

                # Compute loss : change according to multitask

                rec, kl = VAELoss(out_smi, smiles, mu, logv)
                if not use_props:  # VAE only
                    pmse = torch.tensor(0)
                elif use_props :
                    pmse = weightedPropsLoss(p_target, out_p, props_weights)


                val_rec += rec.item()
                val_kl += kl.item()
                val_pmse += pmse.item()

                # Correctly reconstructed characters in first validation batch
                if batch_idx == 0:
                    _, out_chars = torch.max(out_smi.detach(), dim=1)
                    differences = 1. - torch.abs(out_chars - smiles)
                    differences = torch.clamp(differences, min=0., max=1.).double()
                    quality = 100. * torch.mean(differences)
                    quality = quality.detach().cpu()
                    writer.add_scalar('quality/valid', quality.item(), epoch)
                    print('fraction of correct characters at reconstruction // validation : ', quality.item())

            # total Epoch losses
            val_rec, val_kl, val_pmse = val_rec / len(test_loader), \
                                                val_kl / len(test_loader), \
                                                val_pmse / len(test_loader)

            epoch_train_rec, epoch__train_kl, epoch_train_pmse = epoch_train_rec / len(train_loader), \
                                                                             epoch_train_kl / len(train_loader), \
                                                                             epoch_train_pmse / len(train_loader)

        print(f'[Chunk {epoch}], batch valid. loss: rec: {val_rec:.2f}, '
              f'kl: {beta * kl.item():.2f}, props mse: {val_pmse:.2f}')

        # Tensorboard logging
        writer.add_scalar('EpochRec/valid', val_rec, epoch)
        writer.add_scalar('EpochRec/train', epoch_train_rec, epoch)
        writer.add_scalar('EpochKL/valid', val_kl, epoch)
        writer.add_scalar('EpochKL/train', epoch_train_kl, epoch)

        if use_props:
            writer.add_scalar('EpochPropLoss/valid', val_pmse, epoch)
            writer.add_scalar('EpochPropLoss/train', epoch_train_pmse, epoch)
