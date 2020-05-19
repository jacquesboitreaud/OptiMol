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


******* Model finetuning on QSAR training set ***********

"""

import argparse
import sys, os
import torch
import numpy as np

import pickle
import pandas as pd 
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__=='__main__':
    sys.path.append(script_dir)

from utils import ModelDumper, disable_rdkit_logging, setup, log_reconstruction
from dgl_utils import send_graph_to_device
from model import Model
from loss_func import VAELoss, weightedPropsLoss, affsRegLoss, affsClassifLoss
from dataloaders.molDataset import molDataset, Loader
from selfies import decoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--train', help="path to training dataframe", type=str, default='data/finetuning.csv')
    parser.add_argument("--cutoff", help="Max number of molecules to use. Set to -1 for all", type=int, default=-1)

    parser.add_argument('--load_model', action='store_true', default=True)
    parser.add_argument('--load_name', type=str, default='inference_default')  # name of model to load from
    parser.add_argument('--load_iter', type=int, default=1e6)  # resume training at optimize step n°

    parser.add_argument('--decode', type=str, default='selfies')  # 'smiles' or 'selfies'
    parser.add_argument('--build_alphabet', action='store_true', default=False)  # use params.json alphabet

    parser.add_argument('--latent_size', type=int, default=56)  # size of latent code
    parser.add_argument('--n_gcn_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
    
    parser.add_argument('--lr', type=float, default=5e-4)  # Initial learning rate
    parser.add_argument('--clip_norm', type=float, default=50.0)  # Gradient clipping max norm
    parser.add_argument('--beta', type=float, default=0.5)  # initial KL annealing weight
    
    parser.add_argument('--step_beta', type=float, default=0.002)  # beta increase per step
    parser.add_argument('--max_beta', type=float, default=0.5)  # maximum KL annealing weight
    parser.add_argument('--warmup', type=int, default=40000)  # number of steps with only reconstruction loss (beta=0)

    parser.add_argument('--processes', type=int, default=8)  # num workers

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)  # nbr training epochs
    
    parser.add_argument('--anneal_rate', type=float, default=0.9)  # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=40000)  # update learning rate every _ step
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)  # update beta every _ step

    parser.add_argument('--print_iter', type=int, default=1000)  # print loss metrics every _ step
    parser.add_argument('--sample_iter', type=int, default=100)  # print reconstructed smiles every _ step
    
    parser.add_argument('--save_iter', type=int, default=100)  # save model weights every _ step

    # teacher forcing rnn schedule
    parser.add_argument('--tf_init', type=float, default=1.0)
    parser.add_argument('--tf_step', type=float, default=0.002)  # step decrease
    parser.add_argument('--tf_end', type=float, default=0)  # final tf frequency
    parser.add_argument('--tf_anneal_iter', type=int, default=1000)  # nbr of iters between each annealing
    parser.add_argument('--tf_warmup', type=int, default=200000)  # nbr of steps at tf_init

    # Multitask :
    parser.add_argument('--no_props', action='store_true', default=True)  # No multitask props
    parser.add_argument('--no_aff', action='store_true', default=True)  # No multitask aff
    parser.add_argument('--bin_affs', action='store_true', default=False)  # Binned discretized affs or true values

    # =======

    args, _ = parser.parse_known_args()

    logdir, modeldir = setup(args.name, permissive=True)
    dumper = ModelDumper(dumping_path=os.path.join(modeldir, 'params.json'), argparse=args)

    use_props, use_affs = True, True
    if args.no_props:
        use_props = False
    if args.no_aff:
        use_affs = False

    # Multitasking : properties and affinities should be in input dataset
    if use_props:
        properties = ['QED', 'logP', 'molWt']
    else:
        properties = []
    props_weights = [1e3, 1e2, 1]
    if use_affs:
        targets = ['drd3']
    else:
        targets = []
    a_weight = 1e2  # Weight for affinity regression loss

    if args.bin_affs:
        targets = [t + '_binned' for t in targets]  # use binned scores columns
        classes_weights = torch.tensor([0., 1., 1.])  # class weights
        
    # Preload set of actives : 
    actives = pd.read_csv(os.path.join(script_dir,'data', 'finetuning.csv'))
    actives_set = set(list(actives[actives['active']==1].smiles))
    print(len(actives_set))
    
    flag_1, flag_5, flag_10 = 0,0,0

    writer = SummaryWriter(logdir)
    disable_rdkit_logging()  # function from utils to disable rdkit logs

    # Load train set and test set
    loaders = Loader(maps_path='map_files/',
                     csv_path=args.train,
                     vocab=args.decode,
                     build_alphabet=args.build_alphabet,
                     n_mols=args.cutoff,
                     num_workers=args.processes,
                     batch_size=args.batch_size,
                     props=properties,
                     targets=targets)

    train_loader, _, test_loader = loaders.get_data()

    # Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {'features_dim': loaders.dataset.emb_size,  # node embedding dimension
              'num_rels': loaders.num_edge_types,
              'gcn_layers': args.n_gcn_layers,
              'gcn_hdim': 32,
              'l_size': args.latent_size,
              'voc_size': loaders.dataset.n_chars,
              'max_len': loaders.dataset.max_len,
              'N_properties': len(properties),
              'N_targets': len(targets),
              'binned_scores': args.bin_affs,
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
        model.load_no_multitask(weights_path)

    print(model)
    map = ('cpu' if device == 'cpu' else None)
    model.fix_index_to_char()

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print("> learning rate: %.6f" % scheduler.get_lr()[0])

    # Train & test

    if args.load_model:
        total_steps = args.load_iter
    else:
        total_steps = 0
    beta = args.beta
    tf_proba = args.tf_init

    for epoch in range(1, args.epochs + 1):
        print(f'Starting epoch {epoch}')
        model.train()
        epoch_train_rec, epoch_train_kl, epoch_train_pmse, epoch_train_amse = 0, 0, 0, 0

        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(train_loader):

            total_steps += 1  # count training steps

            smiles = smiles.to(device)
            graph = send_graph_to_device(graph, device)
            if use_props:
                p_target = p_target.to(device).view(-1, len(properties))
            if use_affs:
                a_target = a_target.to(device)

            # Forward pass
            mu, logv, _, out_smi, out_p, out_a = model(graph, smiles, tf=tf_proba)

            # Compute loss terms : change according to multitask setting
            rec, kl = VAELoss(out_smi, smiles, mu, logv)

            if not use_affs and not use_props:  # VAE only
                pmse, amse = torch.tensor(0), torch.tensor(0)
            elif use_props and not use_affs:
                pmse = weightedPropsLoss(p_target, out_p, props_weights)
                amse = torch.tensor(0)
            elif use_affs:
                if args.bin_affs:
                    amse = affsClassifLoss(a_target, out_a, classes_weights)
                else:
                    amse = affsRegLoss(a_target, out_a, a_weight)
                if use_props:
                    pmse = weightedPropsLoss(p_target, out_p, props_weights)
                else:
                    pmse = torch.tensor(0)

            # COMPOSE TOTAL LOSS TO BACKWARD
            if total_steps < args.warmup:  # Only reconstruction (warmup)
                t_loss = rec
            else:
                t_loss = rec + beta * kl + pmse + amse

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
                tf = min(args.tf_end, tf_proba - args.tf_step)  # tf decrease

            # logs and monitoring
            if total_steps % args.print_iter == 0:
                print(
                    f'Opt step {total_steps}, rec: {rec.item():.2f}, kl: {beta * kl.item():.2f}, props mse: {pmse.item():.2f}, aff mse: {amse.item():.2f}')
                writer.add_scalar('BatchRec/train', rec.item(), total_steps)
                writer.add_scalar('BatchKL/train', kl.item(), total_steps)
                if use_props:
                    writer.add_scalar('BatchPropMse/train', pmse.item(), total_steps)
                if use_affs:
                    writer.add_scalar('BatchAffMse/train', amse.item(), total_steps)

            if total_steps % args.sample_iter == 0:
                _, out_chars = torch.max(out_smi.detach(), dim=1)
                _, frac_valid = log_reconstruction(smiles, out_smi.detach(),
                                                   loaders.dataset.index_to_char,
                                                   string_type=args.decode)
                print(f'{frac_valid} valid smiles in batch')
                # Correctly reconstructed characters
                differences = 1. - torch.abs(out_chars - smiles)
                differences = torch.clamp(differences, min=0., max=1.).double()
                quality = 100. * torch.mean(differences)
                quality = quality.detach().cpu()
                writer.add_scalar('quality/train', quality.item(), total_steps)
                print('fraction of correct characters at reconstruction : ', quality.item())
                
                #### Sampling step 
                N = 1000
                sample_selfies = []
                with torch.no_grad():
                    samples_z = model.sample_z_prior(n_mols=N)
                    
                    for batch_idx in range(N // 100):
                        batch_z = samples_z[batch_idx * 100:(batch_idx + 1) * 100]
                        gen_seq = model.decode(batch_z)
                        sample_selfies += model.probas_to_smiles(gen_seq)
                    
                    sample_smi = [decoder(s) for s in sample_selfies]
                    
                    actives = [int(s in actives_set) for s in sample_smi]
                    pct_actives = sum(actives)/len(actives)
                    
                    print('PCT ACTIVES in samples : ', pct_actives)
                    
                    if pct_actives >= 0.01 and not flag_1:
                        model.cpu()
                        torch.save(model.state_dict(), os.path.join(modeldir, "weights_1.pth"))
                        model.to(device)
                        flag_1 = True
                        
                    if pct_actives >=0.05 and not flag_5:
                        model.cpu()
                        torch.save(model.state_dict(), os.path.join(modeldir, "weights_5.pth"))
                        model.to(device)
                        flag_5 = True
                        
                    if pct_actives >=0.1 and not flag_10:
                        model.cpu()
                        torch.save(model.state_dict(), os.path.join(modeldir, "weights_10.pth"))
                        model.to(device)
                        flag_10 = True
                        sys.exit()

            # keep track of epoch loss
            epoch_train_rec += rec.item()
            epoch_train_kl += kl.item()
            epoch_train_pmse += pmse.item()
            epoch_train_amse += amse.item()

        # Validation pass
        model.eval()
        val_rec, val_kl, val_amse, val_pmse = 0, 0, 0, 0
        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):

                smiles = smiles.to(device)
                graph = send_graph_to_device(graph, device)

                if use_props:
                    p_target = p_target.to(device).view(-1, len(properties))
                if use_affs:
                    a_target = a_target.to(device)

                mu, logv, z, out_smi, out_p, out_a = model(graph, smiles, tf=tf_proba)

                # Compute loss : change according to multitask

                rec, kl = VAELoss(out_smi, smiles, mu, logv)
                if not use_affs and not use_props:  # VAE only
                    pmse, amse = torch.tensor(0), torch.tensor(0)
                elif use_props and not use_affs:
                    pmse = weightedPropsLoss(p_target, out_p, props_weights)
                    amse = torch.tensor(0)
                elif use_affs:
                    if args.bin_affs:
                        amse = affsClassifLoss(a_target, out_a, classes_weights)
                    else:
                        amse = affsRegLoss(a_target, out_a, a_weight)
                    if use_props:
                        pmse = weightedPropsLoss(p_target, out_p, props_weights)
                    else:
                        pmse = torch.tensor(0)

                val_rec += rec.item()
                val_kl += kl.item()
                val_pmse += pmse.item()
                val_amse += amse.item()

                # Correctly reconstructed characters in first validation batch
                if batch_idx == 0:
                    _, out_chars = torch.max(out_smi.detach(), dim=1)
                    differences = 1. - torch.abs(out_chars - smiles)
                    differences = torch.clamp(differences, min=0., max=1.).double()
                    quality = 100. * torch.mean(differences)
                    quality = quality.detach().cpu()
                    writer.add_scalar('quality/valid', quality.item(), epoch)
                    print('fraction of correct characters in first valid batch : ', quality.item())

            # total Epoch losses
            val_rec, val_kl, val_pmse, t_amse = val_rec / len(test_loader), \
                                                val_kl / len(test_loader), \
                                                val_pmse / len(test_loader), \
                                                val_amse / len(test_loader)

            epoch_train_rec, epoch__train_kl, epoch_train_pmse, epoch_amse = epoch_train_rec / len(train_loader), \
                                                                             epoch_train_kl / len(train_loader), \
                                                                             epoch_train_pmse / len(train_loader), \
                                                                             epoch_train_amse / len(train_loader)

        print(f'[Ep {epoch}/{args.epochs}], batch valid. loss: rec: {val_rec:.2f}, '
              f'kl: {beta * kl.item():.2f}, props mse: {val_pmse:.2f},'
              f' aff mse: {val_amse:.2f}')

        # Tensorboard logging
        writer.add_scalar('EpochRec/valid', val_rec, epoch)
        writer.add_scalar('EpochRec/train', epoch_train_rec, epoch)
        writer.add_scalar('EpochKL/valid', val_kl, epoch)
        writer.add_scalar('EpochKL/train', epoch_train_kl, epoch)

        if use_props:
            writer.add_scalar('EpochPropLoss/valid', val_pmse, epoch)
            writer.add_scalar('EpochPropLoss/train', epoch_train_pmse, epoch)

        if use_affs:
            writer.add_scalar('EpochAffLoss/valid', val_amse, epoch)
            writer.add_scalar('EpochAffLoss/train', epoch_train_amse, epoch)
