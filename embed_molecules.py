# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:13:46 2020

@author: jacqu

Load trained model and use it to embed molecules from their SMILES. 

'baseline.pth' is a plain VAE with multitask prediction of QED, LogP & MolWt
"""

import sys
import numpy as np 
import pandas as pd 


if(__name__=='__main__'):
    sys.path.append('data_processing')
    sys.path.append('dataloaders')
    sys.path.append('eval')
    
    from model_utils import load_trained_model, embed
    from molDataset import molDataset, Loader
    from rdkit_to_nx import smiles_to_nx
    from model import Model 
    
    model_path= f'saved_model_w/baseline.pth'
    
    # Load model on gpu if available
    model, device =load_trained_model(model_path)
    # Initialize dataloader with empty dataset
    dataloader = Loader(test_only=True, num_workers=0)
    
    # Create dataframe with smiles in 'can' column
    smiles = ['C(C[C@@H]1CC[C@H](CC1)Nc2ncccn2)CN3CCN(CC3)c4ccccc4',
              'C(Cc1c[nH]c2ncccc12)N3CCC(=CC3)\C=C\c4ccccc4',
              'C(Cc1c[nH]c2ncccc12)N3CCC(=CC3)c4ccccc4',
              'C(Cc1cc2ccc[nH]c2n1)N3CCC(=CC3)\C=C\c4ccccc4']
    smiles_df = pd.DataFrame.from_dict({'can':smiles})
    
    # embed molecules in latent space of dimension 64
    z = embed(model, device, dataloader, smiles_df) # z has shape (N_molecules, latent_size)