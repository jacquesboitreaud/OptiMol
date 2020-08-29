# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:40:36 2020

@author: jacqu

Split actives and inactives 
"""

import pandas as pd 


df = pd.read_csv('../data/excape_drd3.csv')

df=df[df['active']=='A']

df.to_csv('../data/qsar/actives.csv')


df = pd.read_csv('../data/inactives.csv', usecols = ['SMILES'])

df=df.sample(100000)


df.to_csv('../data/qsar/inactives.csv')