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
import seaborn as sns
import numpy as np
import pandas as pd


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore

    

name = 'clogp_adam_clamp_mid'

norm_scores = True # set to true for clogp

# Plot individual run results 
smiles = plot_csvs(f'../cbas/slurm/results/{name}/docking_results', ylim=(-13.5,-6), plot_best = True, return_best=True, 
                   use_norm_score = norm_scores, obj = 'clogp')
"""
scores = [t[1] for t in smiles]
mols = [Chem.MolFromSmiles(t[0]) for t in smiles]
qeds = np.array([Chem.QED.qed(m) for m in mols])

img=Draw.MolsToGridImage(mols, legends = [f'step {i}: {sc:.2f}, QED = {q:.2f}' for i,(sc,q) in enumerate(zip(scores, qeds))])

    


### Get QED distribution at different steps of CbAS

for step in np.arange(1,31,4):

    samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
    smiles = samples.smile
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    qeds = np.array([Chem.QED.qed(m) for m in mols])
    sns.distplot(qeds, label = f'step {step}', hist = False)
    plt.legend()
"""

# Top mols at step : 
    
save_smiles = {'smile':[]}

step = 20
N_top = 3

samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
samples = samples.sort_values('norm_score')

smiles = samples.smile
scores = samples.norm_score
mols = [Chem.MolFromSmiles(s) for s in smiles]

mols = mols[-N_top:]
scores = scores[-N_top:]

save_smiles['smile']+= list(samples.smile[-N_top:])

img=Draw.MolsToGridImage(mols, molsPerRow= 4, legends = [f'{sc:.2f}, {q:.2f}' for i,(sc,q) in enumerate(zip(scores,qeds))])

soft_mkdir('plots')
img.save(f'plots/cbas_{name}_mols_{step}.png')

df = pd.DataFrame.from_dict(save_smiles)

df.to_csv('clogp_smiles.csv')

#=======


# name = 'clogp_adam_clamp_less'
#
# norm_scores = False # set to true for clogp
#
# # Plot individual run results
# smiles = plot_csvs(f'../cbas/slurm/results/{name}/docking_results', ylim=(-5,20), plot_best = True, return_best=True, use_norm_score = norm_scores)
# scores = [t[1] for t in smiles]
#
# img=Draw.MolsToGridImage([Chem.MolFromSmiles(t[0]) for t in smiles], legends = [f'step {i}: {sc:.2f}' for i,sc in enumerate(scores)])
#
#
#
# ### Get top 3 at last step
# step = 20
# samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}.csv')
# if norm_scores :
#     samples = samples.sort_values('norm_score')
# else:
#     samples = samples.sort_values('score')
# smiles = list(samples[-3:].smile)
# smiles.reverse()
# if norm_scores :
#     scores = list(samples[-3:].norm_score)
# else:
#     scores = list(samples[-3:].score)
#
# scores.reverse()


samples = pd.read_csv(f'carlos/docking/30.csv')
smiles = samples['smile']
scores = samples['score']


list_to_plot = [Chem.MolFromSmiles(s) for s in smiles]
list_qeds = [Chem.QED.qed(m) for m in list_to_plot]
img = Draw.MolsToGridImage(list_to_plot[:20], molsPerRow=7,
                           legends=[f'{sc:.2f}, qed : {qed:.2f}' for i, (sc, qed) in enumerate(zip(scores, list_qeds))])
img.save('TOTOTO.png')

# soft_mkdir('plots')
# img.save(f'plots/cbas_{name}_mols_{step}.png')
