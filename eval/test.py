# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Evaluate training of Graph2Smiles VAE (RGCN encoder, GRU decoder, beam search decoding). 


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

from joblib import dump, load
from sklearn.decomposition import PCA



# Execution is set to take place in graph2smiles root dir 
if (__name__ == "__main__"):
    sys.path.append('eval')
    sys.path.append('dataloaders')
    sys.path.append("data_processing")
    from molDataset import molDataset, Loader
    from rdkit_to_nx import smiles_to_nx
    from model import Model, Loss, RecLoss
    from model_utils import load_trained_model
    
    from eval_utils import *
    from utils import *
    from plot_tools import *
    
    # Eval config 
    model_path= f'saved_model_w/g2s_gpcr.pth'
    
    recompute_pca = False
    reload_model = True
    
    # Should be same as for training...
    properties = ['QED','logP','molWt']
    targets = ['aa2ar','drd3']
    
    # Select only DUDE subset to plot in PCA space 
    plot_target = 'aa2ar'

    #Load eval set: USE MOSES TEST SET !!!!!!!!!!!!!!!!!
    loaders = Loader(csv_path='data/moses_test.csv',
                     n_mols=10,
                     num_workers=0, 
                     batch_size=100, 
                     props = properties,
                     targets=targets,
                     test_only=True,
                     shuffle = True, 
                     select_target = None)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    _, _, test_loader = loaders.get_data()
    
    # Validation pass
    if(reload_model):
         model,device=load_trained_model(model_path)
         model.set_smiles_chars()
         model.eval()
    else:
        try:
            model.eval()
        except NameError:
            model,device=load_trained_model(model_path)
            model.set_smiles_chars()
            model.eval()
    
    # Smiles 
    out_all = np.zeros((loaders.dataset.n,151))
    smi_all = np.zeros((loaders.dataset.n,151))
    
    # Latent embeddings
    z_all = np.zeros((loaders.dataset.n,model.l_size))
    
    # Affinities
    a_target_all = np.zeros((loaders.dataset.n,len(targets)))
    a_all = np.zeros((loaders.dataset.n,len(targets)))
    
    # Mol Props
    p_target_all = np.zeros((loaders.dataset.n,len(properties)))
    p_all = np.zeros((loaders.dataset.n,len(properties)))
    

    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
            
            graph=send_graph_to_device(graph,device)
            
            # Latent embeddings
            z = model.encode(graph, mean_only=True) #z_shape = N * l_size
            pred_p, pred_a = model.props(z), model.affs(z)
            
            # Decoding 
            out = model.decode(z)
            v, indices = torch.max(out, dim=1) # get indices of char with max probability
            
            # Decoding with beam search 
            """
            beam_output = model.decode_beam(z,k=3, cutoff_mols=10)
            # Only valid out molecules 
            mols, _ = log_from_beam(beam_output,loaders.dataset.index_to_char)
            """
            
            # Concat all input and ouput data 
            batch_size = z.shape[0]
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
        # Latent space KDE 
        # ===================================================================
        
        plot_kde(z_all)
        
        # ===================================================================
        # Decoding statistics 
        # ===================================================================
        reconstruction_dataframe, frac_valid, frac_id = log_smiles_from_indices(smi_all, out_all, 
                                                      loaders.dataset.index_to_char)
        print("Fraction of molecules decoded into a valid smiles: ", frac_valid)
        print("Fraction of perfectly reconstructed mols ", frac_id)
        
        # ===================================================================
        # Prediction error plots 
        # ===================================================================
        
        for i,p in enumerate(properties):
            reconstruction_dataframe[p]=p_target_all[:,i]
            reconstruction_dataframe[p+'_pred']=p_all[:,i]
        
        for i,t in enumerate(targets):
            reconstruction_dataframe[t]=-np.log(10e-9*a_target_all[:,i])
            reconstruction_dataframe[t+'_pred']=a_all[:,i]
        reconstruction_dataframe=reconstruction_dataframe.replace([np.inf, -np.inf], 0)
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['QED'],reconstruction_dataframe['QED_pred'])
        sns.lineplot([0,1],[0,1],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['logP'],reconstruction_dataframe['logP_pred'])
        sns.lineplot([-2,5],[-2,5],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe['molWt'],reconstruction_dataframe['molWt_pred'])
        sns.lineplot([0,700],[0,700],color='r')
        
        """
        # Affinities prediction plots
        plt.figure()
        sns.scatterplot(reconstruction_dataframe[targets[0]],reconstruction_dataframe[f'{targets[0]}_pred'])
        sns.lineplot([0,20],[0,20],color='r')
        
        plt.figure()
        sns.scatterplot(reconstruction_dataframe[targets[1]],reconstruction_dataframe[f'{targets[1]}_pred'])
        sns.lineplot([0,20],[0,20],color='r')
        """
        # ===================================================================
        # PCA plot 
        # ===================================================================
        if(recompute_pca):
            fitted_pca = fit_pca(z)
            dump(fitted_pca, 'eval/fitted_pca.joblib') 
            print('Fitted and saved PCA for next time!')
        else:
            try:
                fitted_pca = load('eval/fitted_pca.joblib') 
            except(FileNotFoundError):
                print('Fitted PCA object not found at ~/eval/fitted_pca.joblib, new PCA will be fitted on current data.')
                fitted_pca = fit_pca(z)
                dump(fitted_pca, 'eval/fitted_pca.joblib') 
                print('Fitted and saved PCA for next time!')
        
        # Retrieve affinities of each molecule 
        
        bool1 = [int(a[0]==1) for a in a_target_all] # actives for t1
        
        bit = np.array(bool1)
        bit = [str(i) for i in bit]
        mapping = {'0':'Decoy','1':'Active'}
        bit = [mapping[b] for b in bit]
        bit=pd.Series(bit,index=np.arange(len(bit)))
        
        
        # Plot PCA with desired hue variable 
        plt.figure()
        pca_plot_hue(z= z_all, variable = bit, pca = fitted_pca)
        
        # ====================================================================
        # Random sampling in latent space 
        # ====================================================================
        
        r = torch.tensor(2*np.random.normal(size = z.shape), dtype=torch.float).to('cuda')
        
        out = model.decode(r)
        v, indices = torch.max(out, dim=1)
        indices = indices.cpu().numpy()
        sampling_df, frac_valid,_ = log_smiles_from_indices(None, indices, 
                                                      loaders.dataset.index_to_char)
        
        props, affs = model.props(r).detach().cpu().numpy(), model.affs(r).detach().cpu().numpy()
        
        mols= [Chem.MolFromSmiles(s) for s in list(sampling_df['output smiles'])]
        fig = Draw.MolsToGridImage(mols)
        """
        for m in mols:
            fig = Draw.MolToMPL(m, size = (100,100))
        """