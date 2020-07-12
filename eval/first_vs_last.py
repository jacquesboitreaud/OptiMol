# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:02:16 2020

@author: jacqu
"""

import os, sys
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
script_dir = os.path.dirname(os.path.realpath(__file__))

# OPtimol big model 
df = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/big_new_lr/optimol_scored.csv'))
ordered = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/big_new_lr/ordered.csv'))
scored = {s:sc for s,sc in zip(df.smile,df.score)}
first, last = [], []

for i,s in enumerate(ordered.smile):
    
    if s in scored : 
        if i <=10000:
            first.append(scored[s])
        elif i > 97500:
            last.append(scored[s])

print(len(first), len(last))
plt.figure()
plt.xlim(-14,-4)
sns.distplot(first, label = 'First samples')
sns.distplot(last, label = 'Last samples')
plt.xlabel('Docking score')
plt.ylabel('Density')
plt.legend()
# Multiobj 

df = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/multiobj_big/multiobj_scored.csv'))
ordered = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/multiobj_big/ordered.csv'))
scored = {s:sc for s,sc in zip(df.smile,df.score)}
first, last = [], []

for i,s in enumerate(ordered.smile):
    
    if s in scored : 
        if i <=2500:
            first.append(scored[s])
        elif i >=97500:
            last.append(scored[s])

print(len(first), len(last))
plt.figure()
plt.xlim(-14,-4)
sns.distplot(first, label = 'First samples')
sns.distplot(last, label = 'Last samples')
plt.xlabel('Docking score')
plt.ylabel('Density')
plt.legend()