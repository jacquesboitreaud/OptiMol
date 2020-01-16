# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:28:38 2020

@author: jacqu

Format csv file to contain triplets of molecules, two of them being actives for the selected target
"""

import numpy as np 
import pandas as pd 

df= pd.read_csv('../data/CHEMBL_formatted.csv')
print(df.columns)

# If some cleaning has been done 
# df.to_csv('../data/CHEMBL_formatted.csv')

actives = df[df['HERG']!=0]
decoys = df[df['HERG']==0]

triplets_d = {'can_1':[],'can_2':[],'can_3':[]}
