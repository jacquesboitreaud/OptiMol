# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:27:41 2020

@author: jacqu
Selecting diverse molecules as initial samples; using rdkit MinMax sampler 
"""

import os, sys
import numpy as np
import pandas as pd 
import csv 

from rdkit import Chem 

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


# Load smiles 
with open('../docking/drd3_scores.pickle', 'rb') as f:
    scores = pickle.load(f)
    
smiles = scores.keys()
print(len(smiles))

mols = [Chem.MolFromSmiles(s) for s in smiles]
fps = [GetMorganFingerprint(x,3) for x in mols]

Ntot = len(fps)
Npick = 500

def distij(i,j,fps=fps):
    return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

picker = MaxMinPicker()
pickIndices = picker.LazyPick(distij,Ntot,Npick,seed=23)
pickIndices=list(pickIndices)
    
smiles = [list(smiles)[i] for i in pickIndices]
fps = [fps[i] for i in pickIndices]

# Compute pairwise tanimoto sim 

sim =0
cpt=0
for i in range(Npick):
    for j in range(Npick):
        sim += DataStructs.TanimotoSimilarity(fps[i],fps[j])
        cpt += 1
print('Average tanim sim = ', sim/cpt)

# Save smiles and their scores 



