# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:13:46 2020

@author: jacqu

Load trained model and use it to embed molecules from their SMILES. 


Reconstruction, validity and novelty metrics 


"""

import sys
import numpy as np 
import pandas as pd 
import torch

from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

def valid(smiles):
    # fraction valid smiles 
    return f
    
def exact(smiles_in, smiles_out):
    # fraction of exact reconstruction
    
    return f 
    
def unique(smiles):
    # fraction of unique smiles 
    n = len(smiles)
    nu = len(np.unique(smiles))
    
    return nu/n


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

    #TODO

    # Provide the model with characters corresponding to indices, for smiles generation 
    model.set_smiles_chars(char_file="map_files/zinc_chars.json")
    # Initialize dataloader with empty dataset
    dataloader = Loader(test_only=True, num_workers=0)
    smiles_df=pd.read_csv('data/moses_test.csv')
    smiles_df=smiles_df.sample(1000)
    
    # embed molecules in latent space of dimension 64
    z = model.embed( dataloader, smiles_df) # z has shape (N_molecules, latent_size)
    
    # Decode to smiles 
    decoded = model.decode(torch.tensor(z, dtype=torch.float32).to(device))
    smiles_out = model.probas_to_smiles(decoded)
    
    """
    # Plotting molecules 
    for s in smiles_out : 
        m=Chem.MolFromSmiles(s)
        if(m!=None):
            Draw.MolToMPL(m, size=(120,120))
            plt.show(block=False)
            plt.pause(0.1)
    """