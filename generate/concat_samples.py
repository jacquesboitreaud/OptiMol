# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:32:07 2020

@author: jacqu

Sampling around actives in latent space 
"""

import pandas as pd 
import numpy as np 
import os


if __name__ == "__main__":
    
    d = {'can':[],
         'seed':[]}
    
    batches = os.listdir('outputs')
    batches = [ f for f in batches if '.txt' in f]
    
    for b in batches: 
        seed_n = b[-5]
        with open(f'outputs/{b}','r') as f:
            smiles = f.readlines()
            
        d['can']+=[s.rstrip() for s in smiles]
        d['seed']+= seed_n *len(smiles)
        
df = pd.DataFrame.from_dict(d)
print(f'Dataframe contains {df.shape[0]} samples')
df.to_csv('outputs/act_sampling.csv')