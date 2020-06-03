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


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))
    
from eval.eval_utils import plot_csvs
    

name = 'qed'

# Plot individual run results 
smiles = plot_csvs(f'../cbas/slurm/results/{name}/docking_results', ylim=(0.4,1), plot_best = True, return_best=True)

for i in range(len(smiles)):
    print('Step ', i, smiles[i][0], smiles[i][1])

img=Draw.MolsToGridImage([Chem.MolFromSmiles(t[0]) for t in smiles], legends = [f'step {i}: {t[1]:.2f}' for i,t in enumerate(smiles)])

img.save(f'cbas_{name}_mols.png')
    
