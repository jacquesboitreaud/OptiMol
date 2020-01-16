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
from rdkit_to_nx import *
from molDataset import *

# Print all possible targets to choose from: 
chembls = pd.read_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL_top100.csv')

# CHoose targetsand properties 
targets = ['HERG','Dopamine D3 receptor']
properties = ['QED','logP', 'molWt']
print(len(targets), ' targets selected')


all_c = pd.read_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL.csv')

# ===========================================================================
# Construct set : 
#============================================================================
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
chembl.to_csv('C:/Users/jacqu/Documents/GitHub/graph2smiles/data/CHEMBL_formatted.csv')

# Cleaning step 
#==============================================================================

chembl=pd.read_csv('C:/Users/jacqu/Documents/GitHub/graph2smiles/data/CHEMBL_formatted.csv')

chembl=chembl.reset_index()
# Sanity checks : 
smi = chembl['can']
todrop=[]

char_file = ("C:/Users/jacqu/Documents/GitHub/data/zinc_chars.json")
char_list = json.load(open(char_file))
char_to_index= dict((c, i) for i, c in enumerate(char_list))
char_list = set(char_list)

for i,s in enumerate(smi):
    if(i%100000==0):
        print(i)
    if('9' in s or len(s)>150 or '.' in s):
        todrop.append(i)
        print('.')
    elif('2+' in s or ('3+' in s) or ('2-' in s)):
        print('charge')
        todrop.append(i)
    elif('B' in s):
        for k,c in enumerate(s): 
            if(c=='B' and s[k+1]!='r'):
                todrop.append(i)
                print('B')
    else:
        for c in s : 
            if (c not in char_list):
                todrop.append(i)
                break
            
chembl=chembl.drop(todrop).reset_index()
        
chembl.to_csv('C:/Users/jacqu/Documents/GitHub/graph2smiles/data/CHEMBL_formatted.csv')

# ============================================================================
# Drop by trying to create one-hot vectors (long!)
#=============================================================================

todrop = []
rem, ram, rchim, rcham = loaders.get_maps()
for i,smiles in enumerate(smi):
    if(i%100000==0):
        print(i)
    graph=smiles_to_nx(smiles)
        
    try:
        one_hot = {edge: torch.tensor(rem[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'bond_type')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        
        at_type = {a: oh_tensor(ram[label], len(ram)) for a, label in
                   (nx.get_node_attributes(graph, 'atomic_num')).items()}
        nx.set_node_attributes(graph, name='atomic_num', values=at_type)
        
        at_charge = {a: oh_tensor(rcham[label], len(rcham)) for a, label in
                   (nx.get_node_attributes(graph, 'formal_charge')).items()}
        nx.set_node_attributes(graph, name='formal_charge', values=at_charge)
        
        
        at_chir = {a: torch.tensor(rchim[label]) for a, label in
                   (nx.get_node_attributes(graph, 'chiral_tag')).items()}
        nx.set_node_attributes(graph, name='chiral_tag', values=at_chir)
    except KeyError:
        todrop.append(i)