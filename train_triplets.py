# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE training (RGCN encoder, GRU decoder, teacher forced decoding).
Using triplets loss to disentangle latent space 

****
Default params resume training after 410k iterations, 
starting with 
lr = 1e-5
beta = 0.8 
****
"""

import argparse
import sys
import torch
import numpy as np 
import dgl
import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

import torch.optim.lr_scheduler as lr_scheduler

if (__name__ == "__main__"):
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_path', type=str, default = './saved_model_w/g2s_triplets')
    parser.add_argument('--load_model', type=bool, default=True)
    parser.add_argument('--load_fname', type=str, default='baseline.pth')
    
    parser.add_argument('--use_aff', type=bool, default=False)
    
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=64)
    
    parser.add_argument('--lr', type=float, default=1e-5) # Initial learning rate 
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm 
    parser.add_argument('--beta', type=float, default=0.8) # initial KL annealing weight 
    parser.add_argument('--step_beta', type=float, default=0.002) # beta increase per step 
    parser.add_argument('--max_beta', type=float, default=1.0) # maximum KL annealing weight 
    parser.add_argument('--warmup', type=int, default=0) # number of steps with only reconstruction loss (beta=0)
    
    parser.add_argument('--n_epochs', type=int, default=20) # nbr training epochs 
    parser.add_argument('--anneal_rate', type=float, default=0.9) # Learning rate annealing 
    parser.add_argument('--anneal_iter', type=int, default=40000) # update learning rate every _ step 
    parser.add_argument('--kl_anneal_iter', type=int, default=2000) # update beta every _ step 
    parser.add_argument('--print_iter', type=int, default=50) # print loss metrics every _ step 
    parser.add_argument('--print_smiles_iter', type=int, default=1000) # print reconstructed smiles every _ step 
    parser.add_argument('--save_iter', type=int, default=5000) # save model weights every _ step 
    
    args=parser.parse_args()
    
    sys.path.append("./data_processing")
    from model import Model, Loss, RecLoss, tripletLoss
    from molDataset_3 import Loader
    from utils import *
    
    # config
    n_epochs = args.n_epochs # epochs to train
    batch_size = args.batch_size
    
    properties = ['QED','logP','molWt']
    targets = ['aa2ar','drd3'] # Change target names according to dataset 
    
    model_path= f'saved_model_w/{args.load_fname}'
    log_path='./saved_model_w/logs_g2s.npy'
    
    # Dataloading 
    actives = 'data/triplets/actives_drd3.csv'
    decoys = 'data/triplets/decoys_drd3.csv'

    #Load train set and test set
    loaders = Loader(actives_csv=actives,
                     decoys_csv = decoys,
                     num_workers=0, 
                     batch_size=batch_size, 
                     props = properties,
                     targets=targets)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader = loaders.get_data()
    
    # Logs
    logs_dict = {'train_rec':[],
                 'test_rec':[],
                 'train_pmse':[],
                 'train_simLoss':[],
                 'test_pmse':[],
                 'train_amse':[],
                 'test_amse':[]}
    disable_rdkit_logging() # function from utils to disable rdkit logs
    
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'features_dim':loaders.actives_dataset.emb_size, #node embedding dimension
             'gcn_hdim':128,
             'gcn_outdim':args.latent_size,
             'num_rels':loaders.num_edge_types,
             'l_size':args.latent_size,
             'voc_size':loaders.actives_dataset.n_chars,
             'N_properties':len(properties),
             'N_targets':len(targets),
             'device':device}
    pickle.dump(params, open('saved_model_w/params.pickle','wb'))

    model = Model(**params).to(device)
    if(args.load_model):
        print('Loading pretrained model')
        model.load_state_dict(torch.load(model_path))
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print ("learning rate: %.6f" % scheduler.get_lr()[0])
    
    #Train & test
    #=========================================================================
    model.train()
    total_steps = 0 
    beta = args.beta
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        epoch_rec, epoch_pmse, epoch_simLoss=0,0,0

        for batch_idx, data in enumerate(train_loader):
            
            #Memory debugging function
            #debug_memory()
            total_steps+=1
            
            g_i, s_i, p_i, a_i = data[:4]
            g_j, s_j, p_j, a_j = data[4:8]
            g_l, s_l, p_l, a_l = data[8:]
            
            del(data)
            
            # Pass molecule i 
            p_i=p_i.to(device).view(-1,model.N_properties) # Graph-level target : (batch_size,)
            s_i=s_i.to(device)
            g_i=send_graph_to_device(g_i,device)
            
            p_j=p_j.to(device).view(-1,model.N_properties) # Graph-level target : (batch_size,)
            s_j=s_j.to(device)
            g_j=send_graph_to_device(g_j,device)
            
            p_l=p_l.to(device).view(-1,model.N_properties) # Graph-level target : (batch_size,)
            s_l=s_l.to(device)
            g_l=send_graph_to_device(g_l,device)
            
            mu_i, logv_i, z_i, out_smi_i, out_p_i, out_a_i = model(g_i,s_i)
            mu_j, logv_j, z_j, out_smi_j, out_p_j, out_a_j = model(g_j,s_j)
            mu_l, logv_l, z_l, out_smi_l, out_p_l, out_a_l = model(g_l,s_l)
            
            #Compute loss terms : change according to supervision 
            simLoss = tripletLoss(z_i, z_j, z_l) # Similarity loss for triplet
            
            
            rec_i, kl_i, pmse_i,_= Loss(out_smi_i, s_i, mu_i, logv_i, p_i, out_p_i,\
                                      None, None, train_on_aff=args.use_aff)
            rec_j, kl_j, pmse_j,_= Loss(out_smi_j, s_j, mu_j, logv_j, p_j, out_p_j,\
                                      None, None, train_on_aff=args.use_aff)
            rec_l, kl_l, pmse_l,_= Loss(out_smi_l, s_l, mu_l, logv_l, p_l, out_p_l,\
                                      None, None, train_on_aff=args.use_aff)
            
            rec = rec_i + rec_j + rec_l 
            kl = kl_i + kl_j + kl_l
            pmse = pmse_i + pmse_j + pmse_l
            
            # Deleting loss components after sum
            del([rec_i, rec_j, rec_l, kl_i, kl_j, kl_l, pmse_i, pmse_j, pmse_l])
            
            # COMPOSE TOTAL LOSS TO BACKWARD
            t_loss = rec + kl + pmse + 10e2*simLoss 
            # no affinity loss, beta = 1 from start 
            
            # backward loss 
            optimizer.zero_grad()
            t_loss.backward()
            del(t_loss)
            clip.clip_grad_norm_(model.parameters(),args.clip_norm)
            optimizer.step()
            
            # Annealing KL and LR
            if total_steps % args.anneal_iter == 0:
                 scheduler.step()
                 print ("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_steps % args.kl_anneal_iter == 0 and total_steps >= args.warmup:
                beta = min(1, beta + args.step_beta)
            
            #logs and monitoring
            if total_steps % args.print_iter == 0:
                print('epoch {}, opt. step n°{}, rec_loss {:.2f}, properties mse_loss {:.2f}, \
tripletLoss {:.2f}'.format(epoch, total_steps, rec.item(),pmse.item(), simLoss.item()))
              
            if(total_steps % args.print_smiles_iter == 0):
                reconstruction_dataframe, frac_valid = log_smiles(s_i, out_smi_i.detach(), 
                                                      loaders.actives_dataset.index_to_char)
                print(reconstruction_dataframe)
                print('fraction of valid smiles in training batch: ', frac_valid)
                
            if total_steps % args.save_iter == 0:
                torch.save( model.state_dict(), f"{args.save_path}_iter_{total_steps}.pth")
                pickle.dump(logs_dict, open(f'{log_path}.npy','wb'))
                
            # Add to epoch totals and delete 
            epoch_rec+=rec.item()
            epoch_pmse+=pmse.item()
            epoch_simLoss += simLoss.item()
            
            del(rec)
            del(kl)
            del(pmse)
            del(simLoss)
            
        # End of epoch logs
        epoch_rec, epoch_pmse, epoch_simLoss = epoch_rec/len(train_loader), epoch_pmse/len(train_loader),\
        epoch_simLoss/len(train_loader)
        
        logs_dict['train_pmse'].append(epoch_pmse)
        logs_dict['train_simLoss'].append(epoch_simLoss)
        logs_dict['train_rec'].append(epoch_rec)
        