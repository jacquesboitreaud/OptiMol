# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 17:39:47 2020

@author: jacqu

Train LGBM classifier only on compounds that get correct docking prediction
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

# Get excape docked data 

df = pd.read_csv('../../docking/scores_archive/excape_30k_ex16.csv')

actives = df[df['active']=='A']
inactives = df[df['active']=='N']

# Keep only correctly predicted by docking 

actives = actives[actives['score']<-10]
print('N actives kept:', actives.shape[0])
inactives = inactives[inactives['score']>-8]
print('N inactives kept:', inactives.shape[0])

# Fingerprints : 
n_bits = 2048
smiles = actives.smile

smiles_ok = []
X = []

for i, s in enumerate(smiles):
    m = Chem.MolFromSmiles(s)

    if m != None:
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 3,
                                                   nBits=n_bits)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X.append(np.array(fp).reshape(1, -1))

X = np.concatenate(X, axis=0)
smiles = smiles_ok

# tanimoto distances
D = pairwise_distances(X, metric='jaccard')

distances = []

for i in range(D.shape[0]):
    for j in range(i):
        distances.append(D[i, j])

clusts = ClusterData(data=distances, nPts=D.shape[0], distThresh=0.4, isDistData=True)

nclust = len(clusts)

# sort clusters by size 
clusts = list(clusts)
clusts.sort(key=len)

# Cluster assignment to 5 folds : 
test, valid, train = [], [], []

cpt = 0
folds = [list() for i in range(5)]
while cpt < nclust:

    for i in range(5):
        idces = list(clusts[cpt+i])
        folds[i]+= idces 
    cpt += 5
    
for f in folds :
    print(len(f))

# Array of fingerprints by fold 
X_list = [list() for i in range(5)]
y_list = [list() for i in range(5)]

for i in range(5):
    X_list[i] = X[folds[i]]
    y_list[i] = np.ones(X_list[i].shape[0])
    print('actives in fold', i)
    print(X_list[i].shape)
    print(y_list[i].shape)


# Append inactives ; 

smiles_i = inactives.smile

smiles_ok = []
X_inactives = []
for s in smiles_i:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_inactives.append(np.array(fp).reshape(1,-1))
        
n1=int(len(smiles_ok)/5)
        
# Split inactives into 5 : 
X_inactives_ls = []
y_inactives_ls = []

for i in range(5):
    X_inactives_ls.append(np.array(X_inactives[n1*i:n1*(i+1)]).reshape(n1,2048))
    y_inactives_ls.append(np.zeros(n1))
    print(f'X inactives for fold {i} shape: ',X_inactives_ls[i].shape )
    
# Concatenate actives and inactives into folds 
X_list_full, y_list_full = [], []
for i in range(5):
    X_list_full.append(np.concatenate([X_list[i], X_inactives_ls[i]]) )
    y_list_full.append( np.concatenate([y_list[i], y_inactives_ls[i]]) ) 
    print(f'X inactives + actives for fold {i} shape: ',X_list_full[i].shape )
    
print('>>> Saving list of fold features to pickle')
with open('5folds_excape_docked.pickle','wb') as f:
    pickle.dump(X_list_full,f)
    pickle.dump(y_list_full,f)


