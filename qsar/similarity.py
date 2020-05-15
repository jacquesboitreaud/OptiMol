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

df_a = pd.read_csv('../data/qsar/actives_split.csv')


a_train = df_a[df_a['split']=='train'].smiles
a_test = df_a[df_a['split']=='test'].smiles
a_valid = df_a[df_a['split']=='valid'].smiles

n_bits = 2048

X_train, Y_train = [], []
X_test, Y_test = [], []
X_valid, Y_valid = [], []
smiles_ok = []

for s in a_train:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(1)
        
X_train = np.concatenate(X_train, axis = 0)
Y_train = np.array(Y_train)
smiles =smiles_ok

for s in a_test:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_test.append(np.array(fp).reshape(1,-1))
        Y_test.append(1)
        
X_test = np.concatenate(X_test, axis = 0)
Y_test = np.array(Y_train)
smiles =smiles_ok
      
sims = []
for i in range(X_test.shape[0]):
    minimum = 1
    for j in range(X_train.shape[0]):
        tanim = jaccard(X_test[i], X_train[j])
        if tanim < minimum:
            minimum = tanim
    sims.append(minimum)
    
print('Average similarity to nearest train molecule :', np.mean(sims))
print('Similarity = 1- distance :', 1-np.mean(sims))
    
##############################################################################
############################### Using a random split #########################

df_a = df_a.sample(frac=1).reset_index(drop=True)

a_train = df_a.iloc[:2355, :].smiles
a_test = df_a.iloc[2355:2355+559, :].smiles


X_train, Y_train = [], []
X_test, Y_test = [], []
X_valid, Y_valid = [], []
smiles_ok = []

for s in a_train:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(1)
        
X_train = np.concatenate(X_train, axis = 0)
Y_train = np.array(Y_train)
smiles =smiles_ok

for s in a_test:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_test.append(np.array(fp).reshape(1,-1))
        Y_test.append(1)
        
X_test = np.concatenate(X_test, axis = 0)
Y_test = np.array(Y_train)
smiles =smiles_ok
      
sims_random = []
for i in range(X_test.shape[0]):
    minimum = 0
    for j in range(X_train.shape[0]):
        tanim = jaccard(X_test[i], X_train[j])
        if tanim < minimum:
            minimum = tanim
    sims_random.append(minimum)
    
print('Average distance to nearest train molecule in RANDOM SPLIT :', np.mean(sims_random))
print('Similarity = 1- distance :', 1-np.mean(sims_random))


