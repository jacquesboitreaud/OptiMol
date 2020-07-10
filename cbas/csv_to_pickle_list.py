# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 13:15:35 2020

@author: jacqu

Utils script to read csv with samples and turns into a python list of smiles, then pickle dump. 
"""

import pickle
import pandas as pd 
import os, sys

name = 'to_dock'
df = pd.read_csv(f'slurm/results/{name}/docking_small_results/ordered.csv', header = None )

smiles = list(df.iloc[:,0])

print(len(smiles), ' samples loaded')

N = 2500
# first N 

to_dock_mols = smiles[:N]

# last N 
to_dock_mols+=smiles[-N:]

print(len(to_dock_mols), ' selected for docking')

with open(os.path.join(f'slurm/results/{name}/docking_small_results','docker_samples.p'), 'wb') as f: 
    pickle.dump(smiles, f)
print('Saved smiles to pickle in the same directory')