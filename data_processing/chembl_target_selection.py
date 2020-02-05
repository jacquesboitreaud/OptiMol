# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:23:13 2020

@author: jacqu

Builds chembl dataset for testing separation in model // 
Dataset with 2 targets from CHEMBL; if possible with a significant overlap 
"""

import pandas as pd 
import numpy as np 


df = pd.read_csv('../../data/CHEMBL_top100.csv')
df=df.set_index('target_name')
print(df.index)

t1 = 'Dopamine D3 receptor'

mols = pd.read_csv('../../data/CHEMBL.csv')


# Select molecules corresponding to targets 
mols_1 = mols[mols['target_name']==t1]
mols_1 = mols_1.loc[:,['canonical_smiles','standard_value']]

# duplicates and reformatting 
mols_1 = mols_1.groupby('canonical_smiles').mean().reset_index()
mols_1.columns=['can','t1']
#mols_1['t2']=pd.Series(np.zeros(mols_1.shape[0]), index = mols_1.index)


# concat (if several targets selected)
#mols_a=mols_1.append(mols_2, sort = True)
# remove duplicates if there are 
#mols_a = mols_a.groupby('can').max().reset_index()
    
# Save 
mols_1.to_csv(f'../data/{t1}_chembl.csv')


