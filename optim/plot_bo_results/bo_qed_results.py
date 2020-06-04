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


name = 'qed_100'


threshold=3
    
samples=pd.read_csv(f'../../results/bo/{name}/samples.csv') 

Nsteps = 10 #np.max(samples['step'])

mu, top, stds = [], [], []

for s in range(Nsteps):
    stepsamp=samples[samples['step']==s]
    
    if s==0:
        n_init_samples = stepsamp.shape[0]
    
    mu.append(np.mean(stepsamp.qed))
    stds.append(np.std(stepsamp.qed))
    
    # scores > threshold 
    goods = np.where(stepsamp.qed >= threshold)
    top.append(goods[0].shape[0])

plt.figure()
sns.lineplot(x = np.arange(Nsteps), y=mu)
plt.title(f'Mean score of fresh samples at each step')

plt.figure()
plt.errorbar(np.arange(Nsteps), mu, stds, linestyle='None', marker='^')
plt.xlabel('Step')
plt.ylabel('QED')
plt.show()

plt.figure()
sns.barplot(x = np.arange(Nsteps), y=top, color = 'lightblue')
plt.ylim(0,50)
plt.title(f'Number of samples better than threshold ({threshold}) at each step')

print(f' step 0 contains {n_init_samples} initial samples')