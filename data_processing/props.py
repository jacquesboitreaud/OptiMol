# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

Compute chemical properties for a dataframe
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import pandas as pd

path_to_file = 'C:/Users/jacqu/Documents/GitHub/graph2smiles/data/drd3_chembl.csv'

df=pd.read_csv(path_to_file)
    
smiles = list(df['can'])
d={}
prop_names=['QED','logP','molWt']
for name in prop_names : 
    d[f'{name}']=[]


for i,s in enumerate(smiles) : 
    if(i%1000==0):
        print(i)
    m=Chem.MolFromSmiles(s)
    d['QED'].append(QED.default(m))
    d['logP'].append(Crippen.MolLogP(m))
    d['molWt'].append(Descriptors.MolWt(m))
    

for k in d.keys():
    df[k]=pd.Series(d[k], index = df.index)
    
# Drop lines with Nan properties
df = df.dropna(axis=0)

df.to_csv(path_to_file)
    
