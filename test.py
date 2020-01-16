# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE evaluation (RGCN encoder, GRU decoder, beam search decoding). 


"""
import sys
import torch
import dgl

from rdkit.Chem import Draw

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    from plot_tools import *
    
    # config
    batch_size = 100
    SAVE_FILENAME='./saved_model_w/g2s.pth'
    model_path= 'saved_model_w/g2s.pth'
    
    properties = ['QED','logP','molWt']
    targets = ['t1','t2']

    #Load train set and test set
    loaders = Loader(csv_path='data/validation_2targets.csv',
                     n_mols=1000,
                     num_workers=0, 
                     batch_size=batch_size, 
                     shuffled= False,
                     props = properties,
                     targets=targets,
                     test_only=True)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    _, _, test_loader = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    model = Model(**params).to(device)
    model.load_state_dict(torch.load(model_path))
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Parallel model using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    torch.manual_seed(1)
    
    # Validation pass
    model.eval()
    t_rec, t_mse = 0,0
    z_all = np.zeros((loaders.dataset.n,model.l_size))
    a_target_all = np.zeros((loaders.dataset.n,len(targets)))
    a_all = np.zeros((loaders.dataset.n,len(targets)))
    p_target_all = np.zeros((loaders.dataset.n,len(properties)))
    p_all = np.zeros((loaders.dataset.n,len(properties)))
    
    
    out_all = np.zeros((loaders.dataset.n,151))
    smi_all = np.zeros((loaders.dataset.n,151))
    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
            
            graph=send_graph_to_device(graph,device)
            
            # Latent representations
            z = model.encode(graph, mean_only=True) #z_shape = N * l_size
            pred_p, pred_a = model.props(z), model.affs(z)
            
            # Decoding 
            out = model.decode(z)
            v, indices = torch.max(out, dim=1) # get indices of char with max probability
            
            
            """
            # Decoding to smiles (beam search) and predicted props
            beam_output = model.decode_beam(z,k=3)
            props, aff = model.props(z), model.affs(z)
            # Only valid out molecules 
            mols = log_from_beam(beam_output,loaders.dataset.index_to_char)
            """
            
            # Concat all input and ouput data 
            indices=indices.cpu().numpy()
            out_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=indices
            smi_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=smiles
            
            z=z.cpu().numpy()
            z_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=z
            
            affs = pred_a.cpu().numpy()
            a_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=affs
            
            a_target_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=a_target.numpy()
            
            props = pred_p.cpu().numpy()
            p_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=props
            
            p_target_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=p_target.numpy()
        
        # ===================================================================
        # Decoding statistics 
        # ===================================================================
        reconstruction_dataframe, frac_valid, frac_id = log_smiles_from_indices(smi_all, out_all, 
                                                      loaders.dataset.index_to_char)
        print("Fraction of molecules decoded into a valid smiles: ", frac_valid)
        print("Fraction of perfectly reconstructed mols ", frac_id)
        
        #TODO : properties prediction error + affinities prediction error 
        for i,p in enumerate(properties):
            reconstruction_dataframe[p]=p_target_all[:,i]
            reconstruction_dataframe[p+'_pred']=p_all[:,i]
        
        for i,t in enumerate(targets):
            reconstruction_dataframe[t]=-np.log(10e-9*a_target_all[:,i])
            reconstruction_dataframe[t+'_pred']=a_all[:,i]
        reconstruction_dataframe=reconstruction_dataframe.replace([np.inf, -np.inf], 0)
            
        # ===================================================================
        # Prediction error plots 
        # ===================================================================
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['QED'],reconstruction_dataframe['QED_pred'])
        sns.lineplot([0,1],[0,1],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['logP'],reconstruction_dataframe['logP_pred'])
        sns.lineplot([-2,5],[-2,5],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['molWt'],reconstruction_dataframe['molWt_pred'])
        sns.lineplot([0,700],[0,700],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['t1'],reconstruction_dataframe['t1_pred'])
        sns.lineplot([0,20],[0,20],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['t2'],reconstruction_dataframe['t2_pred'])
        sns.lineplot([0,20],[0,20],color='r')
        
            
        # ===================================================================
        # PCA plot 
        # ===================================================================
        
        bool1 = [int(a[0]!=0) for a in a_all]
        bool2 = [int(a[1]!=0) for a in a_all]
        bit = np.array(bool2)+ 10*np.array(bool1) # bit affinities 
        bit = [str(i) for i in bit]
        mapping = {'0':'dd','1':'da','10':'ad','11':'aa'}
        bit = [mapping[b] for b in bit]
        bit=pd.Series(bit,index=np.arange(len(bit)))
        
        pca_plot_true_affs(z_all,bit)
        
        #TODO : different hue parameters 
        
        # ====================================================================
        # Random sampling in latent space 
        # ====================================================================
        
        r = torch.tensor(np.random.normal(size = z.shape), dtype=torch.float).to('cuda')
        
        out = model.decode(r)
        v, indices = torch.max(out, dim=1)
        indices = indices.cpu().numpy()
        sampling_df, frac_valid,_ = log_smiles_from_indices(None, indices, 
                                                      loaders.dataset.index_to_char)
        
        props, affs = model.props(r).detach().cpu().numpy(), model.affs(r).detach().cpu().numpy()
        
        mols= [Chem.MolFromSmiles(s) for s in list(sampling_df['output smiles'])]
        """
        for m in mols:
            fig = Draw.MolToMPL(m, size = (100,100))
        """