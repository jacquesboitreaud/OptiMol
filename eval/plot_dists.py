# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space // random samples from diagonal normal dist 
Run from repo root. 

"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import argparse
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
from rdkit import Chem

import seaborn as sns
import matplotlib.pyplot as plt 

import pandas as pd 
from rdkit.Chem import Draw, QED
from data_processing.sascorer import calculateScore


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="path to csv from repo root ",
                        default='data/moses_train.csv')
    parser.add_argument('-N', "--n_mols", help="cutoff nbr of mols", type=int, default=10000)
    parser.add_argument( "--smiles_col", help="name of column with smiles", type=str, default='smiles')

    args, _ = parser.parse_known_args()

    df = pd.read_csv(f'../{args.name}', nrows = args.n_mols)

    smiles = df[args.smiles_col] # some csvs (cbas especially) use 'smile' as a header instead. 
    
    qeds, sas = [], []
    
    print(f'>>> computing qed and SA for {args.n_mols} molecules...')
    for s in smiles : 
        
        m= Chem.MolFromSmiles(s)
        if m is not None:
            q = QED.qed(m)
            SA = calculateScore(m)
            
            qeds.append(q)
            sas.append(SA)
            
        
    sns.distplot(qeds, label = 'QED')
    plt.legend()
    plt.xlim(0,1)
    plt.title(args.name)
    plt.figure()
    sns.distplot(sas, label = 'SA_score')
    plt.legend()
    plt.title(args.name)
   
    
    
