# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 15:13:46 2020

@author: jacqu

Load trained model and use it to embed molecules from their SMILES. 


Reconstruction, validity and novelty metrics 


"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import argparse
import networkx as nx

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
from rdkit.Chem import rdmolops

from selfies import decoder

if __name__ == '__main__':
    from eval.eval_utils import *
    from dataloaders.molDataset import  Loader
    from model import  model_from_json
    from data_processing import sascorer

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help="path to csv containing molecules", type=str,
                        default='data/250k_zinc.csv')
    parser.add_argument('-n', "--cutoff", help="Number of molecules to embed. -1 for all", type=int, default=2000)
    
    parser.add_argument('-name', '--name', type=str, default='250k') 

    # =====================
    
    alphabet = '250k_alphabets.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args, _ = parser.parse_known_args()

    # Load model (on gpu if available)
    model = model_from_json(args.name)
    model.to(device)
    model.eval()

    # Load dataframe with mols to embed
    path = os.path.join(script_dir, '../..', args.input)
    if args.cutoff > 0:
        smiles_df = pd.read_csv(path, index_col=0, nrows=args.cutoff)  # cutoff csv at nrows
    else:
        smiles_df = pd.read_csv(path, index_col=0)

    # Initialize dataloader with empty dataset
    dataloader = Loader(props=[], 
                    targets=[], 
                    csv_path = None,
                    maps_path = '../../map_files',
                    alphabet_name = alphabet,
                    vocab='selfies', 
                    num_workers = 0,
                    test_only=True)

    # embed molecules in latent space of dimension 64
    print('>>> Start computation of molecules embeddings...')
    z = model.embed(dataloader, smiles_df)  # z has shape (N_molecules, latent_size)

    # Save molecules latent embeds to pickle.
    savedir = os.path.join(script_dir, '../..', 'data/latent_features_and_targets')
    np.savetxt(os.path.join(savedir, 'latent_features.txt'), z)
    print(f'>>> Saved latent representations of {z.shape[0]} molecules to ~/data/latent_features.txt')

    # Compute properties : 
    smiles_rdkit = smiles_df.smiles
    
    print(f'>>> Computing props for {len(smiles_rdkit)} mols')
    
    logP_values = []
    SA_scores = []
    cycle_scores = []
    
    for i in range(len(smiles_rdkit)):
        logP_values.append(Descriptors.MolLogP(MolFromSmiles(smiles_rdkit[ i ])))
        SA_scores.append(-sascorer.calculateScore(MolFromSmiles(smiles_rdkit[ i ])))

        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(smiles_rdkit[ i ]))))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)
        
        if i % 10000==0:
            print(i)
    
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized
    
    print('>>> Saving targets and split scores to .txt files')
    np.savetxt(os.path.join(savedir, 'targets.txt'), targets)
    np.savetxt(os.path.join(savedir, 'logP_values.txt'), np.array(logP_values))
    np.savetxt(os.path.join(savedir, 'SA_scores.txt'), np.array(SA_scores))
    np.savetxt(os.path.join(savedir, 'cycle_scores.txt'), np.array(cycle_scores))
    print('done!')