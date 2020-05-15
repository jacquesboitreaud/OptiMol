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
df_i = pd.read_csv('../data/qsar/inactives.csv')


n1 = 16666
n2 = 16666*4

smiles_i = df_i.smiles

i_train = smiles_i[:n2]
i_valid = smiles_i[n2:n2+n1]
i_test = smiles_i[n2+n1:]

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
        
for s in i_train:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(0)
        
X_train = np.concatenate(X_train, axis = 0)
Y_train = np.array(Y_train)
smiles =smiles_ok
        
with open('../data/qsar/train.pickle','wb') as f:
    pickle.dump(X_train,f)
    pickle.dump(Y_train,f)
    
# ========= Valid 
    
X_train, Y_train = [], []
X_test, Y_test = [], []
X_valid, Y_valid = [], []
smiles_ok = []
    
for s in a_valid:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(1)
        
for s in i_valid:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(0)
        
X_train = np.concatenate(X_train, axis = 0)
Y_train = np.array(Y_train)
smiles =smiles_ok
        
with open('../data/qsar/valid.pickle','wb') as f:
    pickle.dump(X_train,f)
    pickle.dump(Y_train,f)
    
# =============== Test 
    
X_train, Y_train = [], []
X_test, Y_test = [], []
X_valid, Y_valid = [], []
smiles_ok = []
    
for s in a_test:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(1)
        
for s in i_test:
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        smiles_ok.append(s)
        X_train.append(np.array(fp).reshape(1,-1))
        Y_train.append(0)
        
X_train = np.concatenate(X_train, axis = 0)
Y_train = np.array(Y_train)
smiles =smiles_ok
        
with open('../data/qsar/test.pickle','wb') as f:
    pickle.dump(X_train,f)
    pickle.dump(Y_train,f)

