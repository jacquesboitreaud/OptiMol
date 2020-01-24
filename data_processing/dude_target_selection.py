# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:23:13 2020

@author: jacqu

Builds chembl dataset for testing separation in model // 
Dataset with 2 targets from CHEMBL; if possible with a significant overlap 
"""

import pandas as pd 
import numpy as np 


tar = 'esr1'

mols = pd.read_csv('../data/DUD_clean.csv')

"""
for t in mols.columns:
    if('Unnamed' in t):
        mols=mols.drop(columns=t)

mols.to_csv('../data/DUD_clean.csv')
"""

# Select molecules corresponding to targets 
mols = mols[mols[tar]!=0]
mols = mols.loc[:,['can','QED','logP','molWt',tar]]
    
mols.to_csv(f'../data/dude_targets/{tar}.csv')


