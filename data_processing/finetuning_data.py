# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:58:05 2020

@author: jacqu

Finetuning dataset for VAE with QSAR training set 
"""

import pandas as pd 
from dataloaders.molDataset import molDataset
import numpy as np
from rdkit import Chem
import os, sys

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

df_a = pd.read_csv('../data/qsar/actives_split.csv')
df_i = pd.read_csv('../data/qsar/inactives.csv')


n1 = 16666
n2 = 16666*4

smiles_i = df_i.smiles

i_train = smiles_i[:n2]



# filtering 

dataset = molDataset(csv_path = None, maps_path = '../map_files', vocab='selfies', build_alphabet = False, props=[], targets=[])

"""
dataset.pass_smiles_list(i_train)

inactives = []
for i in range(len(i_train)):
    
    it = dataset.__getitem__(i)
    
    if it[0] is not None :
        
        # kekulize smiles
        s= i_train[i]
        m=Chem.MolFromSmiles(s)
        Chem.Kekulize(m)
        s = Chem.MolToSmiles(m, kekuleSmiles=True)
        inactives.append(s)
        
df= pd.DataFrame.from_dict({'smiles':inactives, 'active': np.zeros(len(inactives))})
"""

# Same for actives     
a_train = df_a[df_a['split']=='train'].smiles
dataset.pass_smiles_list(a_train)

actives = []
for i in range(len(a_train)):
    
    it = dataset.__getitem__(i)
    
    if it[0] is not None :
        
        # kekulize smiles 
        s= a_train[i]
        m=Chem.MolFromSmiles(s)
        Chem.Kekulize(m)
        s = Chem.MolToSmiles(m, kekuleSmiles=True)
        actives.append(s)
        
df2= pd.DataFrame.from_dict({'smiles':actives, 'active': np.ones(len(actives))})

# Save 

df = pd.concat([df,df2]).reset_index(drop=True)
df.to_csv('../data/finetuning.csv')