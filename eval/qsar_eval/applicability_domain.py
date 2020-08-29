# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:57:58 2020

@author: jacqu

Applicability domain for QSAR predictions // SVM 
"""

import pandas 
import numpy as np 

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, precision_score, recall_score
from scipy.spatial.distance import jaccard

import matplotlib.pyplot as plt
import seaborn as sns

import os, sys
from multiprocessing import Pool

import pickle


# 1/ Get QSAR training set features : 
    
with open('../../data/qsar/train.pickle','rb') as f:
    X_train = pickle.load(f)
    y_train = pickle.load(f)
    
# 1b / Set K = sqrt(train set size), the number of closest neighbours considered

X_train = X_train[:7000]

N = X_train.shape[0]
K = int(np.sqrt(N))

print('N=', N)
print('K=', K)

# 2/ Compute pairwise distances 

D = pairwise_distances(X_train, metric='jaccard')

# bonus : PCA to gain speed later. 

# 3/ For each molecule, top K pairwise distances : 
    
mean_dists = []
    
for i in range(N):
    distances_i = sorted(D[i,:])
    distances_i = distances_i[1:K+1] # k closest neighbours
    mean_dist = np.mean(distances_i)
    mean_dists.append(mean_dist)
    
    
mean_dists = np.array(mean_dists)
mu = np.mean(mean_dists)
std = np.std(mean_dists)

print('Max avg distance allowed for applicability:', mu+std)
    

    
    