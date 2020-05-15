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
from sklearn.metrics import pairwise_distances
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

clf = SVC(C=1.0, gamma='auto', probability = True  )

print('>>> Fitting SVM to train molecules')
clf.fit(X_train, y_train)



# ROC Curve 
y_score = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
plt.plot(fpr, tpr)
roc_auc = auc(fpr, tpr)
plt.title(f'Test ROC AUC : {roc_auc:.4f}')

print('QSAR roc AUC :', roc_auc)

with open('qsar_fitted_svm.pickle', 'wb') as f :
    pickle.dump(clf, f)
print('Saved svm to qsar_fitted_svm.pickle')
