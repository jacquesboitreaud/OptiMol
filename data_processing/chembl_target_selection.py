# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:23:13 2020

@author: jacqu

Builds chembl dataset for testing separation in model // 
Dataset with 2 targets from CHEMBL; if possible with a significant overlap 
"""

import pandas as pd 
import numpy as np 

df = pd.read_csv('../data/CHEMBL_top100.csv')
df=df.set_index('target_name')
print(df.index)

t1 = 'HERG'
t2='Dopamine D3 receptor'

mols = pd.read_csv('../data/CHEMBL.csv')


# Select molecules corresponding to targets 
mols_1 = mols[mols['target_name']==t1]
mols_1 = mols_1.loc[:,['canonical_smiles','standard_value']]

mols_2 = mols[mols['target_name']==t2]
mols_2 = mols_2.loc[:,['canonical_smiles','standard_value']]

# duplicates and reformatting 
mols_1 = mols_1.groupby('canonical_smiles').mean().reset_index()
mols_1.columns=['can','t1']
mols_1['t2']=pd.Series(np.zeros(mols_1.shape[0]), index = mols_1.index)

mols_2 = mols_2.groupby('canonical_smiles').mean().reset_index()
mols_2.columns=['can','t2']
mols_2['t1']=pd.Series(np.zeros(mols_2.shape[0]), index = mols_2.index)


# concat 
mols_a=mols_1.append(mols_2, sort = True)
mols_a = mols_a.groupby('can').max().reset_index()


# Add zero props 
properties = ['QED','logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD']
for p in properties:
   mols_a[p]=pd.Series(np.zeros(mols_a.shape[0]), index = mols_a.index)
   
# Sanity checks : 
smi = mols_a['can']
todrop=[]

for i,s in enumerate(smi):
    if('9' in s or len(s)>150):
        todrop.append(i)
    elif ('B' in s and 'Br' not in s):
        todrop.append(i)
    elif('Si' in s):
        todrop.append(i)
    elif('se' in s):
        todrop.append(i)
        print('Se')
    elif('te' in s):
        todrop.append(i)
        print('Te')
mols_a=mols_a.drop(todrop).reset_index()
    
# Save 
mols_a.to_csv('data/validation_2targets.csv')

#mols=pd.read_csv('../data/validation_2targets.csv')


