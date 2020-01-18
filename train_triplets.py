# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE training (RGCN encoder, GRU decoder, teacher forced decoding).

Using triplets loss to disentangle latent space 


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
    parser.add_argument("-m", "--num_triplets", help="number of molecules to use. Set to -1 for all",
                        type=int, default=100)
    parser.add_argument("-e", "--num_epochs", help="number of training epochs", type=int, default=1)
    args=parser.parse_args()
    
    sys.path.append("./data_processing")
    from model import Model, Loss, RecLoss, tripletLoss
    from molDataset_3 import Loader
    from utils import *
    
    # config
    n_epochs = args.num_epochs # epochs to train
    batch_size = 64
    warmup_epochs = 0
    use_aff = False # Penalize error on affinity prediction or not 
    properties = ['QED','logP','molWt']
    targets = ['aa2ar','drd3'] # Change target names according to dataset 
    
    SAVE_FILENAME='./saved_model_w/g2s_triplets.pth'
    model_path= 'saved_model_w/g2s.pth'
    log_path='./saved_model_w/logs_g2s.npy'
    
    load_model = False
    save_model = True
    
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
                 'test_pmse':[],
                 'train_amse':[],
                 'test_amse':[]}
    disable_rdkit_logging() # function from utils to disable rdkit logs
    
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'features_dim':loaders.actives_dataset.emb_size, #node embedding dimension
             'gcn_hdim':128,
             'gcn_outdim':64,
             'num_rels':loaders.num_edge_types,
             'l_size':64,
             'voc_size':loaders.actives_dataset.n_chars,
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
    
    #Train & test
    #=========================================================================
    model.train()
    for epoch in range(1, n_epochs+1):
        print(f'Starting epoch {epoch}')
        epoch_rec, epoch_pmse, epoch_amse=0,0,0

        for batch_idx, data in enumerate(train_loader):
            
            debug_memory()
            
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
            
            z_i, logv_i, out_smi_i, out_p_i, out_a_i = model(g_i,s_i)
            z_j, logv_j, out_smi_j, out_p_j, out_a_j = model(g_j,s_j)
            z_l, logv_l, out_smi_l, out_p_l, out_a_l = model(g_l,s_l)
            
            #Compute loss terms : change according to supervision 
            simLoss = tripletLoss(z_i, z_j, z_l) # Similarity loss for triplet
            print('Triplet loss: ', simLoss)
            
            
            rec_i, kl_i, pmse_i,_= Loss(out_smi_i, s_i, z_i, logv_i, p_i, out_p_i,\
                                      None, None, train_on_aff=use_aff)
            rec_j, kl_j, pmse_j,_= Loss(out_smi_j, s_j, z_j, logv_j, p_j, out_p_j,\
                                      None, None, train_on_aff=use_aff)
            rec_l, kl_l, pmse_l,_= Loss(out_smi_l, s_l, z_l, logv_l, p_l, out_p_l,\
                                      None, None, train_on_aff=use_aff)
            
            rec = rec_i + rec_j + rec_l 
            kl = kl_i + kl_j + kl_l
            pmse = pmse_i + pmse_j + pmse_l
            
            epoch_rec+=rec_i.item() + rec_j.item() + rec_l.item()
            epoch_pmse+=pmse_i.item() + pmse_j.item() + pmse_l.item()
            
            # COMPOSE TOTAL LOSS TO BACKWARD
            t_loss = rec + kl + pmse + simLoss # no affinity loss
            
            # backward loss 
            optimizer.zero_grad()
            t_loss.backward()
            del(t_loss)
            del(rec)
            del(kl)
            del(pmse)
            del(simLoss)
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            
            #logs and monitoring
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, rec_loss {:.2f}, properties mse_loss {:.2f}, \
 sim loss {:.2f}'.format(epoch, 
                      batch_idx, rec.item(),pmse.item(), simLoss.item()))
              
            if(batch_idx==0):
                reconstruction_dataframe, frac_valid = log_smiles(s_i, out_smi_i.detach(), 
                                                      loaders.actives_dataset.index_to_char)
                print(reconstruction_dataframe)
                print('fraction of valid smiles in a training batch: ', frac_valid)

            epoch_rec, epoch_pmse, epoch_amse = epoch_rec/len(train_loader), epoch_pmse/len(train_loader),\
            epoch_amse/len(train_loader)
        
        logs_dict['train_pmse'].append(epoch_pmse)
        logs_dict['train_amse'].append(epoch_amse)
        logs_dict['test_rec'].append(epoch_rec)
            
        if(epoch%2==0 and save_model):
            #Save model : checkpoint      
            torch.save( model.state_dict(), SAVE_FILENAME)
            pickle.dump(logs_dict, open(log_path,'wb'))
            print(f"model saved to {SAVE_FILENAME}")
        