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
import dgl
import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--num_mols", help="number of molecules to use. Set to -1 for all",
                        type=int, default=100)
    parser.add_argument("-e", "--num_epochs", help="number of training epochs", type=int, default=1)
    args=parser.parse_args()
    
    sys.path.append("./data_processing")
    from model import Model, Loss, RecLoss
    from molDataset import molDataset, Loader
    from utils import *
    
    # config
    n_epochs = args.num_epochs # epochs to train
    batch_size = 64
    warmup_epochs = 0
    use_aff = True # Penalize error on affinity prediction or not 
    properties = ['QED','logP','molWt']
    targets = ['t1','t2'] # Change target names according to dataset 
    SAVE_FILENAME='./saved_model_w/g2s_finetune.pth'
    model_path= 'saved_model_w/g2s.pth'
    log_path='./saved_model_w/logs_g2s.npy'
    
    load_model = True
    save_model = True

    #Load train set and test set
    loaders = Loader(csv_path='../data/validation_2targets.csv', # pretraining.csv or finetuning.csv
                     n_mols=args.num_mols,
                     num_workers=0, 
                     batch_size=batch_size, 
                     shuffled= True,
                     props = properties,
                     targets=targets)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader, _, test_loader = loaders.get_data()
    
    # Logs
    logs_dict = {'train_rec':[],
                 'test_rec':[],
                 'train_pmse':[],
                 'test_pmse':[],
                 'train_amse':[],
                 'test_amse':[]}
    disable_rdkit_logging() # function from utils to disable rdkit logs
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'features_dim':loaders.dataset.emb_size, #node embedding dimension
             'gcn_hdim':128,
             'gcn_outdim':128,
             'num_rels':loaders.num_edge_types,
             'l_size':128,
             'voc_size':loaders.dataset.n_chars,
             'N_properties':len(properties),
             'N_targets':len(targets),
             'device':device}
    pickle.dump(params, open('saved_model_w/params.pickle','wb'))

    model = Model(**params).to(device)
    if(load_model):
        model.load_state_dict(torch.load(model_path))
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Start training using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    torch.manual_seed(1)
    optimizer = optim.Adam(model.parameters())
    #optimizer = optim.Adam(model.parameters(),lr=1e-4, weight_decay=1e-5)
    
    #Train & test
    model.train()
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        epoch_rec, epoch_pmse, epoch_amse=0,0,0

        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(train_loader):
            
            p_target=p_target.to(device).view(-1,model.N_properties) # Graph-level target : (batch_size,)
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
            # For warmup epochs, train only on smiles reconstruction 
            if(epoch<warmup_epochs):
                t_loss = rec 
            else: 
                t_loss = rec + kl + pmse + amse
            
            # backward loss 
            optimizer.zero_grad()
            t_loss.backward()
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            
            #logs and monitoring
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, rec_loss {:.2f}, properties mse_loss {:.2f}, \
 aff mse_loss {:.2f}'.format(epoch, 
                      batch_idx, rec.item(),pmse.item(), amse.item()))
              
            if(batch_idx==0):
                reconstruction_dataframe, frac_valid = log_smiles(smiles, out_smi.detach(), 
                                                      loaders.dataset.index_to_char)
                print(reconstruction_dataframe)
                print('fraction of valid smiles in a training batch: ', frac_valid)
        
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
                
            #reconstruction_dataframe = log_smiles(smiles, out, loaders.dataset.index_to_char)
            #print(reconstruction_dataframe)
            
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
        logs_dict['test_rec'].append(epoch_rec)
            
        if(epoch%2==0 and save_model):
            #Save model : checkpoint      
            torch.save( model.state_dict(), SAVE_FILENAME)
            pickle.dump(logs_dict, open(log_path,'wb'))
            print(f"model saved to {SAVE_FILENAME}")
        