# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:02:56 2020

@author: jacqu
"""

import pandas as pd 

df1 = pd.read_csv('../data/moses_scored_valid.csv', usecols = ['smiles', 'drd3'])

df2 = pd.read_csv('../data/excape_drd3_scored.csv', usecols = ['smiles', 'drd3'])

# Sample 1k from each 

df1 = df1.sample(500)
df2=df2.sample(500)

d=pd.concat([df1,df2])

d.to_csv('../data/2k_diverse_samples.csv')