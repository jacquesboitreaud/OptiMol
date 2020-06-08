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

logP_values = np.loadtxt('../data/latent_features_and_targets/logP_values.txt')
SA_scores = np.loadtxt('../data/latent_features_and_targets/SA_scores.txt')
cycle_scores = np.loadtxt('../data/latent_features_and_targets/cycle_scores.txt')

def normalize(m):
    current_log_P_value = logP(m)
    current_SA_score = -calculateScore(m)
    current_cycle_score = -cycle_score(m)
    
    # Normalize 
    current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
    current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
    current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
    
    score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
    
    return score
    

name = 'clogp'

for step in os.listdir(f'../cbas/slurm/results/{name}/docking_results'):
    
    samples = pd.read_csv(f'../cbas/slurm/results/{name}/docking_results/{step}')
    smiles = samples.smile
    scores = []
    
    for i in range(len(smiles)):
        m = Chem.MolFromSmiles(smiles[i])
        if m is not None:
            scores.append(normalize(m))
        else:
            scores.append(-20)
            
    samples['norm_score']=scores
    
    samples.to_csv(f'../cbas/slurm/results/{name}/docking_results/{step}')
    print(f'Normalized and saved scores of {step}')


    
