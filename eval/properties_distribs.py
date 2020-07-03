# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:44:07 2020

@author: jacqu

Compute distributions of QED & SA for samples from a model and training set 
"""

from rdkit import Chem
from rdkit.Chem import Draw, QED
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



## Plot samples in csv  
name = 'shuffled_whole_zinc'

df = pd.read_csv(f'../data/{name}.csv', nrows = 20000)

smiles = df.smiles # some csvs (cbas especially) use 'smile' as a header instead. 

qeds, sas = [], []

for s in smiles : 
    
    m= Chem.MolFromSmiles(s)
    if m is not None:
        q = QED.qed(m)
        SA = calculateScore(m)
        
        qeds.append(q)
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


df = pd.read_csv('../results/saved_models/LN_08sched/samples.csv')

steps = np.arange(10000,370000, 10000)

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
            
                q = Chem.QED.qed(m)
                SA = calculateScore(m)
            
                qeds_training[t].append(q)
                sas_training[t].append(SA)

#sns.distplot(qeds, label = 'QED')
plt.figure()
for t in steps[15:20]: 
    sns.distplot(qeds_training[t], label = f'step_{t}', hist = False)
plt.legend()
plt.xlim(0,1)
plt.title('')
plt.title(name)
