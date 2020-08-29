# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:50:13 2020

@author: jacqu

GBM tree classifier from Target 2 Drug paper 
"""

from lightgbm import LGBMClassifier
import numpy as np 
import pandas as pd 
from sklearn.metrics import pairwise_distances, precision_score, recall_score, f1_score, roc_curve, auc
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.cluster import AgglomerativeClustering

from rdkit.ML.Cluster.Butina import ClusterData

import pickle


def process_one(s):
    m = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
    return fp


# Load data 
with open('5folds_excape_docked.pickle','rb') as f:
    X_list = pickle.load(f)
    y_list = pickle.load(f)

# Model like Target2Drug paper 
model = LGBMClassifier(n_estimators = 1000, colsample_bytree=0.8)


for i in range(5): # crossval 
    
    X = np.concatenate([X_list[j] for j in range(5) if j!=i])
    y = np.concatenate([y_list[j] for j in range(5) if j!=i])
    
    X_out, y_heldout = X_list[i], y_list[i]
    
    print(f'fitting model, fold {i} heldout')
    model.fit(X,y)
    
    # Predict: 
    print(f'running predictions, fold {i} heldout')
    y_pred = model.predict(X_out)
    
    fpr, tpr, thresholds = roc_curve(y_heldout, y_pred)
    roc_auc = auc(fpr, tpr)
    
    y_score = np.round(y_pred)
    prec, rec = precision_score(y_heldout, y_pred), recall_score(y_heldout, y_pred)
    f1 = f1_score(y_heldout, y_pred)
    print('QSAR roc AUC :', roc_auc)
    print('precision :', prec)
    print('recall :', rec)
    print(f'f1 score for fold {i}:', f1)
    
##### Now fit a model on whole dataset : 
    
X = np.concatenate([X_list[j] for j in range(5)])
y = np.concatenate([y_list[j] for j in range(5)])

model.fit(X,y)

with open('../../results/saved_models/lgbm_tolerant.pickle', 'wb') as f:
    pickle.dump(model, f)

