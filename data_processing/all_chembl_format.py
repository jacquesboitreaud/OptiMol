# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 14:22:40 2019

@author: jacqu


Format all chembl dataset with columns corresponding to selected pockets + properties 
"""

import numpy as np
import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED

# Print all possible targets to choose from: 
chembls = pd.read_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL_top100.csv')


# CHoose targets and properties 
targets = ['HERG','Dopamine D3 receptor']
properties = ['QED','logP', 'molWt']
print(len(targets), ' targets selected')


all_c = pd.read_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL.csv')

# Construct set : 
d = {}
for t in targets:
    d[t] = []
for p in properties:
    d[p]=[]
d['can']= [] # column for canonical smiles
    
for i, row in all_c.iterrows(): 
    if(i>590916):
        if(i%10000 == 0):
            print(i)
        target = row['target_name']
        
        # smiles and props
        s=row['canonical_smiles']
        m=Chem.MolFromSmiles(s)
        if(m!=None):
            d['can'].append(s)
            d['QED'].append(QED.default(m))
            d['logP'].append(Crippen.MolLogP(m))
            d['molWt'].append(Descriptors.MolWt(m))
            # targets IC50 
            if(target in targets):
                d[target].append(-np.log(10e9*row['standard_value'])) # pIC50 = -log(IC50 in nanomolar)
                for other in targets:
                    if(other!=target):
                        d[other].append(0)
            else:
                for other in targets:
                    d[other].append(0)
    
chembl = pd.DataFrame.from_dict(d)


# Sanity checks : 
smi = chembl['can']
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
chembl=chembl.drop(todrop).reset_index()
        
chembl.to_csv('C:/Users/jacqu/Documents/GitHub/graph2smiles/data/CHEMBL_formatted.csv')