# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:25:33 2020

@author: jacqu
"""

import pandas as pd 
df=pd.read_csv('data/moses_train.csv', usecols = ['smiles', 'selfies'])

df.to_csv('data/moses_train_light.csv')