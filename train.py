# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

RGCN model to predict molecular LogP (Regression)

Dataset: molecularsets training and test set 
https://github.com/molecularsets/moses/tree/master/data

"""
import sys
import torch
import dgl
import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from model import Model, Loss, RecLoss
    from molDataset import molDataset, Loader
    from utils import *
    
    # config
    n_epochs = 3 # epochs to train
    batch_size = 64
    display_test=False
    SAVE_FILENAME='./saved_model_w/g2s.pth'
    model_path= 'saved_model_w/g2s.pth'
    log_path='./saved_model_w/logs_g2s.npy'
    
    load_model = True

    #Load train set and test set
    loaders = Loader(csv_path='../data/moses_train.csv',
                     n_mols=1000,
                     num_workers=0, 
                     batch_size=batch_size, 
                     shuffled= True,
                     target = 'logP')
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    train_loader, _, test_loader = loaders.get_data()
    
    # Logs
    logs_dict = {'train_rec':[],
                 'test_rec':[],
                 'train_mse':[],
                 'test_mse':[]}
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    params ={'features_dim':loaders.dataset.emb_size, #node embedding dimension
             'gcn_hdim':64,
             'gcn_outdim':64,
             'num_rels':loaders.num_edge_types,
             'l_size':64,
             'voc_size':loaders.dataset.n_chars,
             'N_properties':1,
             'N_targets':1,
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
        epoch_rec, epoch_mse=0,0

        for batch_idx, (graph, smiles, target) in enumerate(train_loader):
            
            target=target.to(device).view(-1,1) # Graph-level target : (batch_size,)
            smiles=smiles.to(device)
            graph=send_graph_to_device(graph,device)
            
            # Forward pass
            out_smi, out_p = model(graph,smiles)
            
            #Compute loss : change according to supervision 
            rec = RecLoss(out_smi, smiles)
            mse = F.mse_loss(out_p, target, reduction='sum')
            epoch_rec+=rec.item()
            epoch_mse+=mse.item()
            
            t_loss = rec + mse
            
            # backward loss 
            optimizer.zero_grad()
            t_loss.backward()
            #clip.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            
            #logs and monitoring
            if batch_idx % 100 == 0:
                # log
                print('ep {}, batch {}, rec_loss {:.2f}, properties mse_loss {:.2f}'.format(epoch, 
                      batch_idx, rec.item(),mse.item()))
              
            if(batch_idx==0):
                reconstruction_dataframe, frac_valid = log_smiles(smiles, out_smi.detach(), 
                                                      loaders.dataset.index_to_char)
                print(reconstruction_dataframe)
                print('fraction of valid smiles in a training batch: ', frac_valid)
        
        # Validation pass
        model.eval()
        t_rec, t_mse = 0,0
        with torch.no_grad():
            for batch_idx, (graph, smiles, target) in enumerate(test_loader):
                
                target=target.to(device).view(-1,1) # Graph-level target : (batch_size,)
                smiles=smiles.to(device)
                graph=send_graph_to_device(graph,device)
                
                
                out_smi, out_p = model(graph,smiles)
            
                #Compute loss : change according to supervision 
                rec = RecLoss(out_smi, smiles)
                mse = F.mse_loss(out_p,target,reduction='sum')
                t_rec += rec.item()
                t_mse += mse.item()
                
            #reconstruction_dataframe = log_smiles(smiles, out, loaders.dataset.index_to_char)
            #print(reconstruction_dataframe)
            
            t_rec, t_mse = t_rec/len(test_loader), t_mse/len(test_loader)
            epoch_rec, epoch_mse = epoch_rec/len(train_loader), epoch_mse/len(train_loader)
                
        print(f'Validation loss at epoch {epoch}, per batch: rec: {t_rec}, mse: {t_mse}')
            
        # Add to logs : 
        logs_dict['test_mse'].append(t_mse)
        logs_dict['test_rec'].append(t_rec)
        logs_dict['train_mse'].append(epoch_mse)
        logs_dict['test_rec'].append(epoch_rec)
            
        if(epoch%2==0):
            #Save model : checkpoint      
            torch.save( model.state_dict(), SAVE_FILENAME)
            pickle.dump(logs_dict, open(log_path,'wb'))
            print(f"model saved to {SAVE_FILENAME}")
        