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
import seaborn as sns

from multiprocessing import Pool

def process_one(s):
    m = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
    return fp

if __name__=='__main__':
    
    # Train test split 
      
    df = pd.read_csv('../data/moses_scored.csv')
    
    
    with open('../results/saved_models/qsar_svm.pickle', 'rb') as f :
        clf = pickle.load(f)
    print('Loaded svm ')
    
    X=[]
    pool=Pool()
    fps = pool.map(process_one, list(df.smiles))
    pool.close()
    X=[np.array(fp).reshape(1, -1) for fp in fps]
    
    X = np.concatenate(X, axis=0)
    
    with open('../data/moses_fps.pickle', 'wb') as f :
        pickle.dump(X,f)
        print('Dumped fps ')
        
    
    # Test metrics 
    print('*********** computing predictions **************')
    y_score = clf.predict_proba(X)[:,1]
    
    df['qsar']=y_score
    
    plt.scatter(x=df['drd3'], y=df['qsar'])
    plt.xlabel('drd3 docking')
    plt.ylabel('qsar score')
    
    positives = df[df['qsar']>=0.5]
    print(positives.shape[0], ' positives')

plt.figure()
sns.distplot(df['qsar'], hist=False)

sns.distplot((df['drd3']-np.max(df['drd3']))/np.min(df['drd3']), hist=False)


