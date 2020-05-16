# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:27:41 2020

@author: jacqu
Selecting diverse molecules as initial samples; using rdkit MinMax sampler 

For QED or docking scores (using the pickle with all previous docking scores)
"""

import os, sys
import numpy as np
import pandas as pd 
import csv 
import argparse

from rdkit import Chem 
from rdkit.Chem import QED

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('-n', "--n_mols", help="Nbr of molecules", type=int, default=500)

args, _ = parser.parse_known_args()

# ======================================



# save_csv 
save_csv = f'../data/diverse_samples.csv'

# Load smiles 
df=pd.read_csv('../data/moses_scored.csv')
smiles = df.smiles
drd3= df.drd3
scores = {}
for i,s in enumerate(smiles) :
    scores[s] = drd3[i]
    
smiles = list(scores.keys())
smiles = [s for s in smiles if len(s)>=8 and scores[s]!=0] # toa avoid non representative and null molecules

print(len(smiles))

mols = [Chem.MolFromSmiles(s) for s in smiles]
fps = [GetMorganFingerprint(x,3) for x in mols]

Ntot = len(fps)
Npick = args.n_mols

def distij(i,j,fps=fps):
    return 1-DataStructs.TanimotoSimilarity(fps[i],fps[j])

picker = MaxMinPicker()
pickIndices = picker.LazyPick(distij,Ntot,Npick,seed=23)
pickIndices=list(pickIndices)
    
smiles = [list(smiles)[i] for i in pickIndices]
mols = [list(mols)[i] for i in pickIndices]
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
affs = [scores[s] for s in smiles]
qed = [QED.qed(m) for m in mols]

header = ['smiles','drd3', 'qed']

with open(save_csv, 'w', newline='') as csvfile:
    csv.writer(csvfile).writerow(header)
    
    for i in range(Npick):
         csv.writer(csvfile).writerow([smiles[i], affs[i], qed[i]])
        
        
    
    



