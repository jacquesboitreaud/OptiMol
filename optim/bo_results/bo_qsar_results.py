# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:12:56 2020

@author: jacqu
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import seaborn as sns
import os, sys
import pandas as pd
from rdkit import Chem


name = 'bo_run_18_0804'
threshold=0.5
    
samples=pd.read_csv(f'../../results/bo/{name}/samples.csv') 

Nsteps = np.max(samples['step'])

mu, top = [], []
for s in range(Nsteps):
    stepsamp=samples[samples['step']==s]
    
    if s==0:
        n_init_samples = stepsamp.shape[0]
        
    if s>0:
        smiles = stepsamp.smiles
        mols = [Chem.MolFromSmiles(str(smi)) for smi in smiles]
        mols =[m for m in mols if m!=None]
        print(len(mols), f' valid samples at step {s}')
    
    mu.append(np.mean(stepsamp.qsar))
    
    # scores > threshold 
    goods = np.where(stepsamp.qsar >= threshold)
    top.append(goods[0].shape[0])

plt.figure()
sns.lineplot(x = np.arange(Nsteps), y=mu)
plt.ylim(0,1)
plt.title(f'Mean score of fresh samples at each step')

plt.figure()
sns.barplot(x = np.arange(Nsteps), y=top, color = 'lightblue')
plt.ylim(0,10)
plt.title(f'Number of samples better than threshold ({threshold}) at each step')

print(f' step 0 contains {n_init_samples} initial samples')