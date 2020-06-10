# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:12:56 2020

@author: jacqu
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import seaborn as sns
import os, sys
import pandas as pd

from rdkit import Chem

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '../..'))

from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore


name = 'clogp_big'
n_runs = 1 

logP_values = np.loadtxt('../../data/latent_features_and_targets/logP_values.txt')
SA_scores = np.loadtxt('../../data/latent_features_and_targets/SA_scores.txt')
cycle_scores = np.loadtxt('../../data/latent_features_and_targets/cycle_scores.txt')

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


threshold=3


for i in range(1,n_runs+1):
    
    samples=pd.read_csv(f'../results/{name}/simulation_{i}/samples.csv') 
    
    Nsteps = 14
    iterations = np.arange(1,Nsteps+1)
    mu, top, stds = [], [], []
    best_scores = []
    
    for s in range(Nsteps):
        stepsamp=samples[samples['step']==s]
        
        # get best mol and score at iter 
        stepsamp= stepsamp.sort_values('clogp')
        row = stepsamp.iloc[-1]
        smile = row.smiles
        score = row.clogp
        m=Chem.MolFromSmiles(smile)
        norm_score = normalize(m)
        best_scores.append(norm_score)
            
        if s==0:
            n_init_samples = stepsamp.shape[0]
        
        mu.append(np.mean(stepsamp.clogp))
        stds.append(np.std(stepsamp.clogp))
        
        # scores > threshold 
        goods = np.where(stepsamp.clogp >= threshold)
        top.append(goods[0].shape[0])
    
    fig, ax = plt.subplots(1, 2)
    mus = np.array(mu)
    stds = np.array(stds)
    ax[0].fill_between(iterations, mus+stds, mus-stds, alpha=.25)
    sns.lineplot(iterations, mus, ax=ax[0])
    ax[0].plot(iterations, best_scores, 'r.')
    sns.despine()
    plt.xlabel('Step')
    ax[0].set_xlim(1,Nsteps+0.5)
    plt.ylabel('cLogP')
    plt.show()
    
    """
    plt.figure()
    sns.barplot(x = np.arange(Nsteps), y=top, color = 'lightblue')
    plt.ylim(0,50)
    plt.title(f'Number of samples better than threshold ({threshold}) at each step')
    """
    
    print(f' step 0 contains {n_init_samples} initial samples')
    
    discovered = samples[samples['step']>0]
    discovered= discovered.sort_values('clogp')
    
    row = discovered.iloc[-1]
    smile = row.smiles
    score = row.clogp
    
    m=Chem.MolFromSmiles(smile)
    norm_score = normalize(m)
    
    print(smile, ' ', norm_score, score)