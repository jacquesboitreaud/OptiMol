# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 22:23:10 2020

@author: jacqu
"""

import pandas as pd 
import os, sys 

script_dir = os.path.dirname(os.path.realpath(__file__))

name = 'multiobj_big'

dockings_dir = os.path.join(script_dir, '..', 'cbas/slurm/results', name, 'samples')

csvs = os.listdir(dockings_dir)
dfs= []

cpt = 0 

for file in csvs :
    
    dfi = pd.read_csv(os.path.join(dockings_dir, file))
    print(dfi.shape[0])
    cpt +=dfi.shape[0]
    dfs.append(dfi)
    
df = pd.concat(dfs)

df.to_csv(os.path.join(script_dir, '..', 'cbas/slurm/results', name, 'samples_scored.csv'))

print(df.shape)
print(cpt)