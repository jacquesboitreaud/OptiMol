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

from selfies import decoder


if __name__=='__main__':

    from dataloaders.molDataset import molDataset, Loader
    from data_processing.rdkit_to_nx import smiles_to_nx
    from model import Model 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input', help="path to csv containing molecules", type=str, default='data/moses_test.csv')
    parser.add_argument('-n', "--cutoff", help="Number of molecules to embed. -1 for all", type=int, default=-1)
    parser.add_argument('-m', '--model', type=str, default='saved_model_w/baseline.pth')
    
    parser.add_argument('-v', '--vocab', type=str, default='selfies')
    parser.add_argument('-d', '--decode', action='store_true', default=True)
    
    # =====================
    
    args=parser.parse_args()
    
    model_path= args.model
    
    # Load model (on gpu if available)
    params = pickle.load(open('saved_model_w/model_params.pickle','rb')) # model hparams
    model = Model(**params)
    model.load(model_path)
    device = model.device
    # Provide the model with characters corresponding to indices, for smiles generation 
    model.eval()
    
    # Load dataframe with mols to embed
    if(args.cutoff>0):
        smiles_df=pd.read_csv(args.input, index_col=0,  nrows = args.cutoff) # cutoff csv at nrows
    else:
        smiles_df=pd.read_csv(args.input, index_col=0)
        
    # Initialize dataloader with empty dataset
    dataloader = Loader(maps_path='map_files/',
                     vocab = args.vocab,
                     build_alphabet = False,
                     n_mols=args.cutoff,
                     num_workers=0,
                     test_only=True,
                     batch_size=min(32, smiles_df.shape[0]),
                     props = [],
                     targets=[])
    
    # embed molecules in latent space of dimension 64
    print('>>> Start computation of molecules embeddings...')
    z = model.embed( dataloader, smiles_df) # z has shape (N_molecules, latent_size)
    
    # Save molecules latent embeds to pickle.
    with open('data/latent_rpz.pickle','wb') as f:
        pickle.dump(z,f)
    print(f'>>> Saved latent representations of {z.shape[0]} molecules to ~/latent_rpz.pickle')
    
    # Decode to smiles 
    if(args.decode):
        
        if args.vocab == 'smiles':
            print('>>> Decoding to smiles')
            decoded = model.decode(torch.tensor(z, dtype=torch.float32).to(device))
            smiles_out = model.probas_to_smiles(decoded)
            for s in smiles_out :
                print(s)
                
        elif args.vocab == 'selfies' : 
            print('>>> Decoding to selfies')
            decoded = model.decode(torch.tensor(z, dtype=torch.float32).to(device))
            selfies = model.probas_to_smiles(decoded) 
            smiles = [decoder(s) for s in selfies]
            
            for s in smiles :
                print(s)
            
            
    
    """
    # Plotting molecules 
    for s in smiles_out : 
        m=Chem.MolFromSmiles(s)
        if(m!=None):
            Draw.MolToMPL(m, size=(120,120))
            plt.show(block=False)
            plt.pause(0.1)
    """