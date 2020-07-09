# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:23 2020

@author: jacqu

QED distribution of CbAS samples 
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

mus, stds = [], []
top_mus, top_stds = [], []

for step in np.arange(1,steps+1):

    samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
    samples = samples.sort_values('score')
    N = int(samples.shape[0]*percentage_cutoff)
    
    smiles = samples.smile
    smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    values = np.array([Chem.QED.qed(m) for m in mols])
    
    mu, std = np.mean(values), np.std(values)
    
    top_values = values[:N]
    
    top_mu, top_std = np.mean(top_values), np.std(top_values)
    
    mus.append(mu)
    stds.append(std)
    top_mus.append(top_mu)
    top_stds.append(top_std)
    print(f'step {step} processed')
    
mus = np.array(mus)
stds=np.array(stds)
top_mus = np.array(top_mus)
top_stds = np.array(top_stds)

iterations = np.arange(1,steps+1)
ax = plt.gca()
plt.ylim(0,1)
ax.fill_between(iterations, mus + stds, mus - stds, alpha=.25)
sns.lineplot(iterations, mus, ax=ax)

ax.fill_between(iterations, top_mus + top_stds,top_mus - top_stds, alpha=.25, color = 'r')
sns.lineplot(iterations, top_mus, ax=ax, color = 'r')

