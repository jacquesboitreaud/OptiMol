# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 08:24:47 2019

@author: jacqu

DATA CLEANING : 
    
    - Filtering RDKit bugs at computing charges / properties
    - Removing SMILES with 9 cycles or longer than maximum length set
"""

import numpy as np
import pandas as pd

def smiles_cleaning(df):
    # Removes SMILES longer than the threshold
    smiles = list(df['can'])
    todrop=[]
    
    for i, smi in enumerate(smiles):
        if('9' in smi or len(smi)>150 or 'p' in smi):
            todrop.append(i)
    print(f'gonna drop {len(todrop)} bad indices. Shape is ', df.shape)
    df=df.drop(todrop)
    print('Dropped. Shape now is ', df.shape)
    return df 


    
    
if(__name__=='__main__'):
    
    df= pd.read_csv('../data/pretraining.csv')
    df=smiles_cleaning(df)
    df.to_csv('../data/pretraining.csv')
