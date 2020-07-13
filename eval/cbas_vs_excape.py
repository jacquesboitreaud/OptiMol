# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:23 2020

@author: jacqu

Find actives similar to CbAS hits // a-posteriori validation on experimental actives 
"""

import os
import sys

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from scipy.spatial.distance import jaccard


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore

    

name = 'big_new_lr'
step = 15 # The CBAS step to analyze 
topN = 10 # find closest for top ... samples

# Fps of ExCAPE actives 

actives = pd.read_csv(os.path.join(script_dir,'..','data','excape_drd3.csv'))
actives = actives[actives['active']=='A']

a_smiles = actives.smile
print(f'{len(a_smiles)} actives from ExCAPE loaded')

a_smiles = [s for s in a_smiles if Chem.MolFromSmiles(s) is not None]
a_mols = [Chem.MolFromSmiles(s) for s in a_smiles]
a_qeds = [Chem.QED.qed(m) for m in a_mols]

a_fps = [AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048) for m in a_mols]

a_fps = np.array(a_fps)

# ===========================================

samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
samples = samples.sort_values('score')

# QEDs
all_smiles = samples.smile
mols = [Chem.MolFromSmiles(s) for s in all_smiles]
qeds = [Chem.QED.qed(m) for m in mols]
qeds_tops = qeds[:100]
# plot comparative with excape
plt.xlim(0,1)
sns.distplot(a_qeds, color = 'g', label = 'excape actives')
sns.distplot(qeds, label = f'step {step}')
sns.distplot(qeds_tops, color='r', label = f'Top 100 at step {step}')
plt.legend()


# Finding close actives to top samples 

samples = samples[:topN]
smiles = samples.smile
smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
mols = [Chem.MolFromSmiles(s) for s in smiles]
fps = [AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048) for m in mols]

fps= np.array(fps)

# Pairwise distances with actives fps 

closest = []
for i in range(fps.shape[0]):
    maximum = 0
    closest_active = 0
    print(f'>>> finding closest active to sample {i}')
    for j in range(a_fps.shape[0]):
        tanim = 1-jaccard(fps[i], a_fps[j]) # similarity
        if tanim > maximum:
            maximum = tanim
            closest_active = j
    closest.append((smiles[i],a_smiles[closest_active],tanim))
    print('tanimoto sim : ', tanim)
    
imgs = []
for tup in closest : 
    sample_m = Chem.MolFromSmiles(tup[0])
    active_m = Chem.MolFromSmiles(tup[1])
    pair = [sample_m, active_m]
    imgs.append(Draw.MolsToGridImage(pair, molsPerRow= 2, legends = ['sample', f'active, sim = {tup[2]:.2f}']))

idx_to_show = 4
imgs[idx_to_show]


