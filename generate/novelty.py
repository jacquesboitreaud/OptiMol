# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:32:29 2020

@author: jacqu

Check the novelty of the generated molecules and find the closest in train set 
"""

import pandas as pd 
import numpy as np 

import pybel 
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# 1/ Load training molecules and generated batch, filter to get 'new' molecules

train = pd.read_csv('../../data/moses_train.csv')

finetune = pd.read_csv('../../data/exp/gpcr/gpcr.csv')
finetune = finetune[finetune['fold']==1]

seen = set(train['can'])
seen.update(finetune['can'])

# Load generated 

gen = list(np.load('sampled_batch_1.npy',allow_pickle=True))

new = 0
smiles = []
for g in gen : 
    if(g in seen):
        pass
    else:
        smiles.append(g)
        new+=1
np.save('new_mols', smiles)
        
# 2/ Fingerprints 
        
smiles = list(np.load('new_mols.npy', allow_pickle=True))

mols=[]
todrop=[]
for i,s in enumerate(smiles): 
    try: 
        mols.append(pybel.readstring("smi", s))
    except:
        pass
        todrop.append(i)
smiles = [s for i,s in enumerate(smiles) if i not in todrop]
fps = [x.calcfp() for x in mols]

print (fps[0] | fps[1])

for s in smiles : 
    m=Chem.MolFromSmiles(s)
    if(m!=None):
        Draw.MolToMPL(m, size=(120,120))
        plt.show(block=False)
        plt.pause(0.1)
    
    