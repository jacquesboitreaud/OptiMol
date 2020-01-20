# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:47:13 2020

@author: jacqu

Prepare dataset for JTVAE benchmark training 
"""

import pandas as pd 

df = pd.read_csv('../data/DUD_clean.csv')

with open('../data/DUD_clean.txt', 'w') as f:
    f.write(df['can'].str.cat(sep='\n'))
    
