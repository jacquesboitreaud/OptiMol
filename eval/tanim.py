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

N=20 
name = 'q07_bis'

for i in range(1,N):

    df_i = pd.read_csv(f'plot/{name}/{i}.csv')
    
    n_bits = 2048
    smiles = df_i.smile
    
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
    
    m = 1-np.mean(D)
    
    print(m)
