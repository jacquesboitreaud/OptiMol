# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Evaluate training of Graph2Smiles VAE (RGCN encoder, GRU decoder, beam search decoding). 

Open multitask affinity model and test it 

Plot all diagnostic plots 


"""
import os
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import torch
import dgl

from rdkit.Chem import Draw
from selfies import decoder

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances, silhouette_score


import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from joblib import dump, load
from sklearn.decomposition import PCA



# Execution is set to take place in graph2smiles root dir 
if __name__ == "__main__":
    from dataloaders.molDataset import molDataset, Loader
    from data_processing.rdkit_to_nx import smiles_to_nx
    from model import Model, Loss, multiLoss
    
    from eval.eval_utils import *
    from utils import *
    
    # Eval config 
    model_path= f'../saved_model_w/aff_model_iter_320000.pth'
    
    recompute_pca = False
    reload_model = True
    
    # Should be same as for training
    properties = ['QED','logP','molWt']
    targets = ['drd3_binned']
    N = 1000
    
    # Select only DUDE subset to plot in PCA space 
    plot_target = 'drd3'

    #Load eval set: USE MOSES TEST SET !!!!!!!!!!!!!!!!!
    loaders = Loader(csv_path='../data/moses_test.csv',
                     maps_path= '../map_files/',
                     n_mols=N ,
                     vocab = 'selfies',
                     num_workers=0, 
                     batch_size=100, 
                     props = properties,
                     targets=targets,
                     test_only=True)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    _, _, test_loader = loaders.get_data()
    
    # Validation pass
    if(reload_model):
        params = pickle.load(open('../saved_model_w/model_params.pickle','rb'))
        model = Model(**params)
        device = model.load(model_path, aff_net=False)
        model.eval()
    else:
        try:
            model.eval()
        except NameError:
            params = pickle.load(open('../saved_model_w/params.pickle','rb'))
            model = Model(**params)
            device = model.load(model_path)
            model.set_smiles_chars()
            model.eval()
    
    # Smiles 
    out_all = np.zeros((loaders.dataset.n,loaders.dataset.max_len))
    smi_all = np.zeros((loaders.dataset.n,loaders.dataset.max_len))
    
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
            
            props = pred_p.cpu().numpy()
            p_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=props
            
            p_target_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=p_target.numpy()
            
            # Classification of binned affinities
            _, c = torch.max(pred_a, dim=1)
            affs = c.cpu().numpy().reshape(-1,1)
            a_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=affs
            
            a_target_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=a_target.numpy()
            
            
        # ===================================================================
        # Latent space KDE 
        # ===================================================================
        
        plot_kde(z_all)
        
        D_a = pairwise_distances(z_all, metric='cosine')
        print('Average cosine distance in latent space : ', np.mean(D_a))
        
        # ===================================================================
        # Decoding statistics 
        # ===================================================================
        reconstruction_dataframe, frac_valid, frac_id = log_smiles_from_indices(smi_all, out_all, 
                                                      loaders.dataset.index_to_char)
        #only for smiles 
        #print("Fraction of molecules decoded into a valid smiles: ", frac_valid)
        #print("Fraction of perfectly reconstructed mols ", frac_id)
        smiles = [decoder(s) for s in reconstruction_dataframe['output smiles']]
        
        # ===================================================================
        # Prediction error plots 
        # ===================================================================
        
        for i,p in enumerate(properties):
            reconstruction_dataframe[p]=p_target_all[:,i]
            reconstruction_dataframe[p+'_pred']=p_all[:,i]
        
        for i,t in enumerate(targets):
            reconstruction_dataframe[t]=a_target_all[:,i]
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
        sns.lineplot([-12,-5],[-12,-5],color='r')
        
        MAE = np.mean(np.abs(reconstruction_dataframe[targets[0]]-reconstruction_dataframe[f'{targets[0]}_pred']))
        print(f'MAE = {MAE} kcal/mol')
        """
        
        # ===================================================================
        # PCA plot 
        # ===================================================================
        if(recompute_pca):
            fitted_pca = fit_pca(z)
            dump(fitted_pca, '../eval/fitted_pca.joblib') 
            print('Fitted and saved PCA for next time!')
        else:
            try:
                fitted_pca = load('../eval/fitted_pca.joblib') 
            except(FileNotFoundError):
                print('Fitted PCA object not found at ~/eval/fitted_pca.joblib, new PCA will be fitted on current data.')
                fitted_pca = fit_pca(z)
                dump(fitted_pca, 'eval/fitted_pca.joblib') 
                print('Fitted and saved PCA for next time!')
        
        
        # Plot PCA with desired hue variable 
        plt.figure()
        pca_plot_hue(z= z_all, variable = p_target_all[:,1], pca = fitted_pca)
        
        # PCA Affinities
        plt.figure()
        pca_plot_hue(z= z_all, variable = a_all[:,0], pca = fitted_pca)
        
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
        fig = Draw.MolsToGridImage(mols)
        """
        for m in mols:
            fig = Draw.MolToMPL(m, size = (100,100))
        """