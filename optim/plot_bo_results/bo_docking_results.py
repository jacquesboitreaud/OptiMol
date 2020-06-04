# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:12:56 2020

@author: jacqu
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import seaborn as sns
import pandas as pd


name = 'docking_100'

df= pd.read_csv(f'../../results/bo/{name}/samples.csv')
    
    
### Get ZINC scores distribution 
with open(f'../../data/drd3_scores.pickle', 'rb') as f:
    zinc_scores = pickle.load(f)
zinc_scores = np.array(list(zinc_scores.values()))
    
N = np.max(df.step)
means, stds = [], []
good_samples = []

for i in range(N):
    
    scores = df[df['step']==i].aff
    mu = np.mean(scores)
    means.append(mu)
    stds.append(np.std(scores))
    
    idces = np.where(np.array(scores).flatten()>=10.0)
    N_good = idces[0].shape[0]
    good_samples.append(N_good)
    
    plt.figure()
    plt.xlim(-12,0)
    sns.distplot(-scores, label = f'step {i}', kde = False, norm_hist = True)
    sns.distplot(zinc_scores, label = f'zinc', kde = True, hist = False, color = 'green')
  
plt.figure()
sns.lineplot(x = np.arange(N), y=means)
plt.xlabel('Step')
plt.ylabel('Docking scores')

plt.figure()
plt.errorbar(np.arange(N), means, stds, linestyle='None', marker='^')
plt.xlabel('Step')
plt.ylabel('Docking scores')
plt.show()

plt.figure()
sns.barplot(x = np.arange(N), y=good_samples, color = 'lightblue')
plt.title('Number of samples <= -10 at each step')