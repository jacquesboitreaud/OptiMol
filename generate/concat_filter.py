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
    finetune = pd.read_csv('../data/exp/offtarget/herg_drd.csv')
    finetune = finetune[finetune['fold'] == 1]

    seen = set(train['can'])
    seen.update(finetune['can'])
    
    print(len(seen), ' non-novel molecules loaded to filter')
    
    
    batches = os.listdir('outputs')
    batches = [ f for f in batches if '.txt' in f]
    
    for b in batches: 
        seed_n = int(b[:-8])
        print(seed_n)
        with open(f'outputs/{b}','r') as f:
            smiles = f.readlines()
            
        smiles = [s.rstrip() for s in smiles if s not in seen]
        #print(len(smiles))
        d['can']+=smiles
        d['seed']+= [seed_n  for i in range(len(smiles))]
        #print(len(d['seed']))
        #print(len(d['can']))
        
    df = pd.DataFrame.from_dict(d)
    print(f'Dataframe contains {df.shape[0]} samples')
    
    # cutoff at 60 max per seed ; 
    l=[]
    for i in range(0,100):
        dfs = df[df['seed']==i]
        print(dfs.shape[0])
        if(dfs.shape[0]>60):
            dfs=dfs.sample(60)
        l.append(dfs)
        
    D = pd.concat(l)

    D=D.reset_index(drop=True)
    
    writepath = 'outputs/act_sampling_ft.csv'
    D.to_csv(writepath)
    print(f'wrote df to {writepath}')
    print('contains {D.shape[0]} samples')