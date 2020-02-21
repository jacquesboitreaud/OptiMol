# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:32:07 2020

@author: jacqu

Sampling around actives in latent space 

Concat and filter molecules from text files in outputs 
"""

import pandas as pd 
import numpy as np 
import os


if __name__ == "__main__":
    
    d = {'can':[],
         'seed':[]}
    
    train = pd.read_csv('../data/moses_train.csv')    
    finetune = pd.read_csv('../data/exp/herg_drd.csv')
    finetune = finetune[finetune['fold'] == 1]

    seen = set(train['can'])
    seen.update(finetune['can'])
    
    print(len(seen), ' non-novel molecules loaded to filter')
    
    
    batches = os.listdir('outputs')
    batches = [ f for f in batches if '.txt' in f]
    
    for b in batches: 
        seed_n = b[:-8]
        with open(f'outputs/{b}','r') as f:
            smiles = f.readlines()
            
        smiles = [s.rstrip() for s in smiles if s not in seen]
        d['can']+=smiles
        d['seed']+= seed_n *len(smiles)
        
    df = pd.DataFrame.from_dict(d)
    print(f'Dataframe contains {df.shape[0]} samples')
    df.to_csv('outputs/act_sampling_baseline.csv')