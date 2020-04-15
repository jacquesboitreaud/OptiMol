# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Compute selfies for all smiles 
"""
import pandas as pd 

try: 
    import selfies
    from selfies import encoder, decoder, selfies_alphabet
except:
    print('Please install selfies package by running "pip install selfies" ')
    

train = pd.read_csv('moses_train.csv', nrows = 10, index_col = 0)

for s in train.iloc[:,0]:
    
    print('smiles :', s)
    print('SELFIES : ', encoder(s))