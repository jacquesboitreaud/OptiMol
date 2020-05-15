# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:27:36 2020

@author: jacqu

QSAR SVM for active / inactive binary prediction 

Like in latentGAN : ECFP6 (= Morgan radius 3 in rdkit), 2048 Bits fps 
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd 
import pickle

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, precision_score, recall_score
from scipy.spatial.distance import jaccard

from sklearn.metrics.pairwise import rbf_kernel

import matplotlib.pyplot as plt

      
# Train test split 
  
with open('../data/qsar/train.pickle','rb') as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)
    
with open('../data/qsar/test.pickle','rb') as f:
    X_test= pickle.load(f)
    y_test = pickle.load(f)
    
with open('../data/qsar/valid.pickle','rb') as f:
    X_valid = pickle.load(f)
    y_valid= pickle.load(f)

# Subsample the training set 
p = 0.2

n_inactives_train = y_train.shape[0]-2355
ni = int(p*n_inactives_train)

X_train, y_train = X_train[:2355+ni], y_train[:2355+ni]

print(f'>>> training on {y_train.shape[0]} samples')

c_list = [32, 64, 128]
gamma_list = [ 1e-2, 5e-2 ]
results= {}


for gamma in gamma_list:

    # Computing kernel 
    kernel_train = rbf_kernel(X_train, gamma=gamma)
    kernel_valid = rbf_kernel(X_train, X_valid, gamma=gamma)
    kernel_valid = np.transpose(kernel_valid)
    
    # Fitting SVM 
    
    for c in c_list:
    
        clf = SVC(C=c, gamma='auto', kernel='precomputed', probability = True  )
        
        print('>>> Fitting SVM to train molecules')
        clf.fit(kernel_train, y_train)
        
        
        # ROC Curve 
        y_score = clf.predict_proba(kernel_valid)[:,1]
        
        
        fpr, tpr, thresholds = roc_curve(y_valid, y_score)
        roc_auc = auc(fpr, tpr)
        y_score = np.round(y_score)
        precision, rec = precision_score(y_valid, y_score), recall_score(y_valid, y_score)
        print('QSAR roc AUC :', roc_auc)
        print('precision :', precision)
        print('recall :', rec)
        
        print(f'QSAR roc AUC  (c={c}, gamma = {gamma}):', roc_auc)
        print(f'QSAR f1 score  (c={c}, gamma = {gamma}):', 2*(precision*rec)/(precision+rec))
        
        # add to dict 
        results[(c,gamma)]=roc_auc
        
lines = []
for c in c_list:
    line = []
    for gamma in gamma_list:
        line.append(results[(c,gamma)])
    lines.append(line)

import csv
save_csv = '../results/grid_search.csv'
header = ['C\gamma']+gamma_list
with open(save_csv, 'w', newline='') as csvfile:
    csv.writer(csvfile).writerow(header)
    
    for i,c in enumerate(c_list):
        csv.writer(csvfile).writerow([c]+lines[i])

