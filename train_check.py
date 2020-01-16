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
    batch_size = 128
    warmup_epochs = 0
    use_aff = True # Penalize error on affinity prediction or not 
    properties = ['QED','logP','molWt']
    #targets = np.load('map_files/targets_chembl.npy')[:2]
    targets = ['HERG','Dopamine D3 receptor']
    SAVE_FILENAME='./saved_model_w/g2s.pth'
    model_path= 'saved_model_w/g2s.pth'
    log_path='./saved_model_w/logs_g2s.npy'
    
    load_model = True
    save_model = True

    #Load train set and test set
    loaders = Loader(csv_path='data/CHEMBL_formatted.csv', # pretraining.csv or finetuning.csv
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
            
            pass
            
        
        # Validation pass
        model.eval()
        t_rec, t_amse, t_pmse = 0,0,0
        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
                
                pass
        