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


# Fitting SVM 
    
C, gamma = 32.0, 0.01

clf = SVC(C=C, gamma=gamma, probability = True  )

print(f'>>> Fitting SVM to train molecules: C = {C}, gamma = {gamma}')
clf.fit(X_train, y_train)

with open('qsar_fitted_svm.pickle', 'wb') as f :
    pickle.dump(clf, f)
print('Saved svm to qsar_fitted_svm.pickle')


# Train metrics  
print('************Train set **************')
y_score = clf.predict_proba(X_train)[:,1]

fpr, tpr, thresholds = roc_curve(y_train, y_score)
roc_auc = auc(fpr, tpr)

y_score = np.round(y_score)
precision, rec = precision_score(y_train, y_score), recall_score(y_train, y_score)

print('QSAR roc AUC :', roc_auc)
print('precision :', precision)
print('recall :', rec)



# Valid metrics 
print('************Validation set **************')
y_score = clf.predict_proba(X_valid)[:,1]

fpr, tpr, thresholds = roc_curve(y_valid, y_score)
roc_auc = auc(fpr, tpr)

y_score = np.round(y_score)
precision, rec = precision_score(y_valid, y_score), recall_score(y_valid, y_score)


print('QSAR roc AUC :', roc_auc)
print('precision :', precision)
print('recall :', rec)


# Test metrics 
print('*********** Test set **************')
y_score = clf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

y_score = np.round(y_score)
precision, rec = precision_score(y_test, y_score), recall_score(y_test, y_score)

print('QSAR roc AUC :', roc_auc)
print('precision :', precision)
print('recall :', rec)



