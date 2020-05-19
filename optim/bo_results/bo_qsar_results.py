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


name = 'qsar_300'
threshold=0.5
    
samples=pd.read_csv(f'../../results/bo/{name}/samples.csv') 

Nsteps = 200 #np.max(samples['step'])

mu, top, stds = [], [], []
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
    stds.append(np.std(stepsamp.qsar))
    
    # scores > threshold 
    goods = np.where(stepsamp.qsar >= threshold)
    top.append(goods[0].shape[0])

plt.figure()
sns.lineplot(x = np.arange(Nsteps), y=mu)
plt.ylim(0,1)
plt.xlabel(f'Step')
plt.ylabel(f'QSAR score')

plt.figure()
sns.barplot(x = np.arange(Nsteps), y=top, color = 'lightblue')
plt.ylim(0,10)
plt.title(f'Number of samples better than threshold ({threshold}) at each step')

plt.figure()
plt.errorbar(np.arange(Nsteps), mu, stds, linestyle='None', marker='^')
plt.xlabel('Step')
plt.ylabel('QSAR scores')
plt.show()

print(f' step 0 contains {n_init_samples} initial samples')

discovered = samples[samples['step']>0]
discovered = discovered[discovered['qsar']>threshold]

print(discovered)