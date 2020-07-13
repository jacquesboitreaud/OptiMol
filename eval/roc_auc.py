# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:44:07 2020

@author: jacqu

Compute ROC AUC for ZINC vs Gianni, or ZINC vs OptiMol (Gianni metric)
"""

from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score

import sys, os 

script_dir = os.path.dirname(os.path.realpath(__file__))

## Load data 
    
optimol = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'optimol_samples_scored.csv'))
optimol['subset']=1

zinc = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'zinc_docked.csv'))
zinc['subset']=0

df=pd.concat([optimol,zinc])
df=df.sort_values('score')

### Enrichment : 

auc = roc_auc_score(y_true = df['subset'], y_score = -df['score']) # binary labels ; reverse the scores (greater scores for 'positives')

print('OptiMol auc :', auc)
    

# Gianni AUC 

gianni = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'fabritiis_docked.csv'))
gianni['subset']=1

zinc = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'zinc_docked.csv'))
zinc['subset']=0

df=pd.concat([gianni,zinc])
df=df.sort_values('score')

### Enrichment : 

auc = roc_auc_score(y_true = df['subset'], y_score = -df['score']) # binary labels ; reverse the scores (greater scores for 'positives')

print('De Fabritiis auc :', auc)