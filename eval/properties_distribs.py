# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:44:07 2020

@author: jacqu

Compute distributions of QED & SA for samples from a model and training set 
"""

from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import pandas as pd
import numpy as np

import os, sys
import matplotlib.pyplot as plt 
import seaborn as sns

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))
    
from data_processing.sascorer import calculateScore



## Training set 
name = 'step_10'

df = pd.read_csv(f'../data/{name}.csv', nrows = 20000)

smiles = df.smile

qeds, sas = [], []

for s in smiles : 
    
    m= Chem.MolFromSmiles(s)
    if m is not None:
        QED = Chem.QED.qed(m)
        SA = calculateScore(m)
        
        qeds.append(QED)
        sas.append(SA)
    

## Samples 

with open('../data/gen.txt', 'r') as f :
    smiles = f.readlines()
    smiles = [s.rstrip() for s in smiles]
    
name = 'model_samples'

sas_samples = []
qeds_samples = []

for s in smiles : 
    
    m= Chem.MolFromSmiles(s)
    
    if m is not None:
    
        QED = Chem.QED.qed(m)
        SA = calculateScore(m)
    
        qeds_samples.append(QED)
        sas_samples.append(SA)
    
sns.distplot(qeds, label = 'QED')
sns.distplot(qeds_samples, label = 'QED_samples')
plt.legend()
plt.title(name)
plt.figure()
sns.distplot(sas, label = 'SA_score')
sns.distplot(sas_samples, label = 'SA_score_samples')
plt.legend()
plt.title(name)
    

# Samples taken during training : 


df = pd.read_csv('../results/saved_models/ln_sched_big/samples.csv')

steps = np.arange(211000,225000, 2000)

qeds_training , sas_training = {}, {}

for t in steps : 
    
    df_t = df[df['step']==t]
    
    smiles = df_t.smiles
    qeds_training[t] = []
    sas_training[t] = []
    
    for s in smiles : 
        
        if type(s)==str:
            m= Chem.MolFromSmiles(s)
            
            if m is not None:
            
                QED = Chem.QED.qed(m)
                SA = calculateScore(m)
            
                qeds_training[t].append(QED)
                sas_training[t].append(SA)

sns.distplot(qeds, label = 'QED')
for t in steps: 
    sns.distplot(qeds_training[t], label = f'step_{t}', hist = False)
plt.legend()
plt.title(name)
