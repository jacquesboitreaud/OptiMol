# -*- coding: utf-8 -*-
"""
Created on Thu May 14 16:44:20 2020

@author: jacqu
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

from rdkit.ML.Cluster.Butina import ClusterData

import csv 

df = pd.read_csv('../data/qsar/actives.csv')

n_bits = 2048
smiles = df.smile

smiles_ok = []
X=[]

for i,s in enumerate(smiles):
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X.append(np.array(fp).reshape(1,-1))
        
X = np.concatenate(X, axis = 0)
smiles =smiles_ok


# tanimoto distances 
D = pairwise_distances(X, metric = 'jaccard')
with open('../data/qsar/pairwise_dists.pickle', 'wb') as f :
    pickle.dump(D,f)

distances = []

for i in range(D.shape[0]):
    for j in range(i):
        distances.append(D[i,j])

clusts = ClusterData(data = distances, nPts = D.shape[0], distThresh =0.4, isDistData=True  )

nclust = len(clusts)

# sort clusters by size 
clusts = list(clusts)
clusts.sort(key=len)

# Cluster assignment to train, valid and test : like in olivecrona 
test, valid, train = [], [], []

cpt=0
while cpt  < nclust:
    
    for i in range(6):
        idces = list(clusts[cpt+i])
        if i ==0:
            test+= idces
        elif i==1:
            valid+=idces
        else:
            train+=idces
    cpt+=6
    
print(len(test))
print(len(train))
print(len(valid))

N = D.shape[0]

header = ['smiles', 'split']
save_csv = '../data/qsar/actives_split.csv'

with open(save_csv, 'w', newline='') as csvfile:
    csv.writer(csvfile).writerow(header)

    with open(save_csv, 'a', newline='') as csvfile:
        for i in train:
            csv.writer(csvfile).writerow([smiles[i], 'train'])
        for i in test:
            csv.writer(csvfile).writerow([smiles[i], 'test'])
        for i in valid:
            csv.writer(csvfile).writerow([smiles[i], 'valid'])

    
    

    
    
    
    
    
    