# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:23 2020

@author: jacqu

Internal diversity of CbAS samples 
"""

import os
import sys

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import jaccard

from sklearn.metrics import pairwise_distances


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore

    

name = 'multiobj_qed4' # run name 
steps = 30

percentage_cutoff = 0.1 # compute intdiv on top 10% molecules in the samples 


### Get QED distribution at different steps of CbAS

tanim_dist_all, tanim_dist_top = [],[]

for step in np.arange(1,steps+1):

    samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
    samples = samples.sort_values('score')
    N = int(samples.shape[0]*percentage_cutoff)
    
    smiles = samples.smile
    smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048) for m in mols]
    
    fps= np.array(fps)
    
    D= pairwise_distances(fps, metric = 'jaccard')
    D_top = D[:N,:N]
    
    tanim_dist_all.append(np.mean(D))
    tanim_dist_top.append(np.mean(D_top))
    
sns.lineplot(x=np.arange(1, step+1), y=tanim_dist_all, color = 'b', label = 'all samples')
sns.lineplot(x=np.arange(1, step+1), y=tanim_dist_top, color = 'r', label = f'top {percentage_cutoff*100:.0f}%')
plt.ylim(0,1)
plt.ylabel('Average fingerprint pairwise distance')
plt.xlabel('Step')

