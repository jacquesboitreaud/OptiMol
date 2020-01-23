# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE training (RGCN encoder, GRU decoder, teacher forced decoding). 


"""

import argparse
import sys
import torch
import numpy as np 

import pickle
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help="path to training dataframe", type=str, default='data/DUD_clean.csv')
    parser.add_argument("--cutoff", help="Max number of molecules to use. Set to -1 for all", type=int, default=100)
    parser.add_argument('--save_path', type=str, default = './saved_model_w/g2s')
    parser.add_argument('--load_model', type=bool, default=False)
    
    parser.add_argument('--use_aff', type=bool, default=False)
    
    parser.add_argument('--hidden_size', type=int, default=450)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=64)
    
    parser.add_argument('--lr', type=float, default=1e-3) # Initial learning rate 
    parser.add_argument('--clip_norm', type=float, default=50.0) # Gradient clipping max norm 
    parser.add_argument('--beta', type=float, default=0.0) # initial KL annealing weight 
    parser.add_argument('--step_beta', type=float, default=0.002) # beta increase per step 
    parser.add_argument('--max_beta', type=float, default=1.0) # maximum KL annealing weight 
    parser.add_argument('--warmup', type=int, default=40000) # number of steps with only reconstruction loss (beta=0)
    
    parser.add_argument('--n_epochs', type=int, default=20) # nbr training epochs 
    parser.add_argument('--anneal_rate', type=float, default=0.9) # Learning rate annealing 
    parser.add_argument('--anneal_iter', type=int, default=40000) # update learning rate every _ step 
    parser.add_argument('--kl_anneal_iter', type=int, default=2000) # update beta every _ step 
    parser.add_argument('--print_iter', type=int, default=100) # print loss metrics every _ step 
    parser.add_argument('--print_smiles_iter', type=int, default=1000) # print reconstructed smiles every _ step 
    parser.add_argument('--save_iter', type=int, default=10000) # save model weights every _ step 
    
     # =======
    
    args=parser.parse_args()
    
    sys.path.append("./data_processing")
    from model import Model, Loss, RecLoss
    from molDataset import molDataset, Loader
    from utils import *
    
    # config
    properties = ['QED','logP','molWt']
    targets = ['aa2ar','drd3'] # Change target names according to dataset 
    
    
    load_path= 'saved_model_w/g2s'
    log_path='./saved_model_w/logs_g2s'
    load_model = args.load_model
    
    n_epochs = args.n_epochs 
    batch_size = args.batch_size
    use_aff = args.use_aff  
    

    #Load train set and test set
    loaders = Loader(csv_path=args.train,
                     n_mols=args.cutoff,
                     num_workers=0, 
                     batch_size=args.batch_size, 
                     props = properties,
                     targets=targets)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader, _, test_loader = loaders.get_data()
    
    # Logs
    disable_rdkit_logging() # function from utils to disable rdkit logs
    logs_dict = {'train_rec':[],
                 'test_rec':[],
                 'train_pmse':[],
                 'test_pmse':[],
                 'train_amse':[],
                 'test_amse':[]}
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'features_dim':loaders.dataset.emb_size, #node embedding dimension
             'gcn_hdim':128,
             'gcn_outdim':64,
             'num_rels':loaders.num_edge_types,
             'l_size':args.latent_size,
             'voc_size':loaders.dataset.n_chars,
             'N_properties':len(properties),
             'N_targets':len(targets),
             'device':device}
    pickle.dump(params, open('saved_model_w/params.pickle','wb'))

    model = Model(**params).to(device)
    if(load_model):
        model.load_state_dict(torch.load(f'{load_path}.pth'))
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    
    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print ("learning rate: %.6f" % scheduler.get_lr()[0])
    
    #Train & test
    model.train()
    total_steps=0
    beta = 0 
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        epoch_rec, epoch_pmse, epoch_amse=0,0,0

        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(train_loader):
            
            total_steps+=1 # count training steps 
            
            p_target=p_target.to(device).view(-1,model.N_properties)
            a_target=a_target.to(device).view(-1,model.N_targets)
            smiles=smiles.to(device)
            graph=send_graph_to_device(graph,device)
            
            # Forward pass
            mu, logv, out_smi, out_p, out_a = model(graph,smiles)
            
            #Compute loss terms : change according to supervision 
            rec, kl, pmse, amse= Loss(out_smi, smiles, mu, logv, p_target, out_p,\
                                      a_target, out_a, train_on_aff=use_aff)
            epoch_rec+=rec.item()
            epoch_pmse+=pmse.item()
            epoch_amse += amse.item()
            
            # COMPOSE TOTAL LOSS TO BACKWARD
            if(total_steps<args.warmup):
                t_loss = rec 
            else: 
                t_loss = rec + beta*kl + pmse + amse
            
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
aff mse_loss {:.2f}'.format(epoch, total_steps, rec.item(),pmse.item(), amse.item()))
              
            if(total_steps % args.print_smiles_iter == 0):
                reconstruction_dataframe, frac_valid = log_smiles(smiles, out_smi.detach(), 
                                                      loaders.dataset.index_to_char)
                print(reconstruction_dataframe)
                print('fraction of valid smiles in training batch: ', frac_valid)
                
            if total_steps % args.save_iter == 0:
                torch.save( model.state_dict(), f"{args.save_path}_iter_{total_steps}.pth")
                pickle.dump(logs_dict, open(f'{log_path}.npy','wb'))

        # Validation pass
        model.eval()
        t_rec, t_amse, t_pmse = 0,0,0
        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
                
                p_target=p_target.to(device).view(-1,model.N_properties)
                a_target = a_target.to(device).view(-1,model.N_targets)
                smiles=smiles.to(device)
                graph=send_graph_to_device(graph,device)
                
                mu, logv, out_smi, out_p, out_a = model(graph,smiles)
            
                #Compute loss : change according to supervision 
                rec, kl, p_mse, a_mse = Loss(out_smi, smiles, mu, logv,\
                           p_target, out_p, a_target, out_a, train_on_aff=use_aff)
                t_rec += rec.item()
                t_pmse += p_mse.item()
                t_amse += a_mse.item()
            
            t_rec, t_pmse, t_amse = t_rec/len(test_loader), t_pmse/len(test_loader), t_amse/len(test_loader)
            epoch_rec, epoch_pmse, epoch_amse = epoch_rec/len(train_loader), epoch_pmse/len(train_loader),\
            epoch_amse/len(train_loader)
                
        print(f'Validation loss at epoch {epoch}, per batch: rec: {t_rec:.2f}, props mse: {t_pmse:.2f},\
 aff mse: {t_amse:.2f}')
            
        # Add to logs : 
        logs_dict['test_pmse'].append(t_pmse)
        logs_dict['test_amse'].append(t_amse)
        logs_dict['test_rec'].append(t_rec)
        
        logs_dict['train_pmse'].append(epoch_pmse)
        logs_dict['train_amse'].append(epoch_amse)
        logs_dict['train_rec'].append(epoch_rec)
        