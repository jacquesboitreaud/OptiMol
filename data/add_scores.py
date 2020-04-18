# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:48:10 2020

@author: jacqu

Add docking scores column to moses_train dataset.
Molecules with no score get 0.0 score. All other scores are negative (interact° energy)
"""

import pandas as pd
import pickle
with open('drd3_scores.pickle', 'rb') as f :
    scored_smiles = pickle.load(f)

df = pd.read_csv('moses_train.csv', index_col= 0)

smiles = df.smiles 

scores = []

print('>>> Adding scores to moses_train dataset')
for s in smiles :
    if s in scored_smiles : 
        scores.append(scored_smiles[s])
    else:
        scores.append(0)
        
df['drd3'] = pd.Series(scores, index = df.index)

df.to_csv('moses_train.csv')
print('Saved train subset with "drd3" column. Value 0.0 indicates no data')

# test set 
df = pd.read_csv('moses_test.csv', index_col= 0)
smiles = df.smiles
scores = []

print('>>> Adding scores to moses_test dataset')
for s in smiles :
    if s in scored_smiles : 
        scores.append(scored_smiles[s])
    else:
        scores.append(0)
        
df['drd3'] = pd.Series(scores, index = df.index)
df.to_csv('moses_test.csv')
print('Saved test subset with "drd3" column. Value 0.0 indicates no data')
