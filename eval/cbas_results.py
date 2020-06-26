# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:23 2020

@author: jacqu

Plotting cbas results 
"""

import os 
import sys

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))
    
from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore
    

name = 'clogp_adam_clamp_less'

norm_scores = False # set to true for clogp

# Plot individual run results 
smiles = plot_csvs(f'../cbas/slurm/results/{name}/docking_results', ylim=(-5,20), plot_best = True, return_best=True, use_norm_score = norm_scores)
scores = [t[1] for t in smiles]

img=Draw.MolsToGridImage([Chem.MolFromSmiles(t[0]) for t in smiles], legends = [f'step {i}: {sc:.2f}' for i,sc in enumerate(scores)])
    


### Get top 3 at last step 
step = 20
samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
if norm_scores :
    samples = samples.sort_values('norm_score')
else:
    samples = samples.sort_values('score')
smiles = list(samples[-3:].smile)
smiles.reverse()
if norm_scores :
    scores = list(samples[-3:].norm_score)
else:
    scores = list(samples[-3:].score)

scores.reverse()

img=Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in smiles], molsPerRow= 1, legends = [f'{sc:.2f}' for i,sc in enumerate(scores)])


soft_mkdir('plots')
img.save(f'plots/cbas_{name}_mols_{step}.png')

    
