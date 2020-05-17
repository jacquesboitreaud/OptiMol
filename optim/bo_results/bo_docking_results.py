# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:12:56 2020

@author: jacqu
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import seaborn as sns


name = 'aff_1'

df= pd.read_csv(f'../../results/bo/{name}/samples.csv')
    
    
### Get ZINC scores distribution 
with open(f'../../data/drd3_scores.pickle', 'rb') as f:
    zinc_scores = pickle.load(f)
zinc_scores = np.array(list(zinc_scores.values()))
    
N = len(scores)
means = []
good_samples = []

for i in range(N):
    
    scores = df[df['step']==i].drd3
    mu = np.mean(scores)
    means.append(mu)
    
    idces = np.where(scores.flatten()>=10.0)
    N_good = idces[0].shape[0]
    good_samples.append(N_good)
    
    plt.figure()
    plt.xlim(-12,0)
    sns.distplot(-scores, label = f'step {i}', kde = False, norm_hist = True)
    sns.distplot(zinc_scores, label = f'zinc', kde = True, hist = False, color = 'green')
  
plt.figure()
sns.lineplot(x = np.arange(N), y=means)
plt.title('Mean score of fresh samples at each step')

plt.figure()
sns.barplot(x = np.arange(N), y=good_samples, color = 'lightblue')
plt.title('Number of samples <= -10 at each step')