# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:28:38 2020

@author: jacqu

Format csv file to contain triplets of molecules, two of them being actives for the selected target
"""

import numpy as np 
import pandas as pd 

df= pd.read_csv('../data/train_gpcr.csv')
print(df.columns)

actives = df[df['drd3']==1]
actives=actives.reset_index(drop=True)
decoys = df[df['drd3']==-1]
decoys=decoys.reset_index(drop=True)

actives.to_csv('data/triplets/actives_drd3.csv')
decoys.to_csv('data/triplets/decoys_drd3.csv')

# If some cleaning has been done 
# df.to_csv('../data/CHEMBL_formatted.csv')

actives = df[df['HERG']>7]
decoys = df[df['HERG']==0]

triplets_d = {'can_1':[],'can_2':[],'can_3':[]}
