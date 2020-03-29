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
import pickle
import argparse


from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt


if __name__=='__main__':

    from dataloaders.molDataset import molDataset, Loader
    from data_processing.rdkit_to_nx import smiles_to_nx
    from model import Model 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-csv', '--csv', help="path to csv containing molecules", type=str, default='data/moses_test.csv')
    parser.add_argument('-n', "--cutoff", help="Number of molecules to embed. -1 for all", type=int, default=10)
    parser.add_argument('-m', '--model', type=str, default='saved_model_w/baseline.pth')
    
    parser.add_argument('-d', '--decode', action='store_true', default=False)
    
    # =====================
    
    args=parser.parse_args()
    
    model_path= args.model
    
    # Load model (on gpu if available)
    params = pickle.load(open('saved_model_w/params.pickle','rb')) # model hparams
    model = Model(**params)
    model.load(model_path)
    device = model.device
    # Provide the model with characters corresponding to indices, for smiles generation 
    model.set_smiles_chars(char_file="map_files/zinc_chars.json") 
    model.eval()

    # Initialize dataloader with empty dataset
    dataloader = Loader(maps_path='map_files/',
                        test_only=True, num_workers=0)
    
    # Load dataframe with mols to embed
    if(args.cutoff>0):
        smiles_df=pd.read_csv(args.csv, nrows = args.cutoff) # cutoff csv at nrows
    else:
        smiles_df=pd.read_csv(args.csv)
    
    # embed molecules in latent space of dimension 64
    print('>>> Start computation of molecules embeddings...')
    z = model.embed( dataloader, smiles_df) # z has shape (N_molecules, latent_size)
    
    # Save molecules latent embeds to pickle.
    with open('latent_rpz.pickle','wb') as f:
        pickle.dump(z,f)
    print(f'>>> Saved latent representations of {z.shape[0]} molecules to ~/latent_rpz.pickle')
    
    # Decode to smiles 
    if(args.decode):
        print('>>> Decoding to smiles')
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