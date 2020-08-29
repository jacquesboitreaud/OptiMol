# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:27:36 2020

@author: jacqu

Check correlation of QSAR with docking oracle. 
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

import os, sys
from multiprocessing import Pool

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '../..'))

def process_one(s):
    m = Chem.MolFromSmiles(s)
    if m is not None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
        return fp
    else:
        return np.zeros(2048)

if __name__=='__main__':
    
    model = 'lgbm_tolerant'
    with open(f'../../results/saved_models/{model}.pickle', 'rb') as f : # lgbm.pickle or qsar_svm.pickle
        model = pickle.load(f)
    print('Loaded model ')
    
    
    df = pd.read_csv('../../docking/scores_archive/excape_30k_ex16.csv')
    print(f'>>> computing fps for samples')

    X=[]
    pool=Pool()
    fps = pool.map(process_one, list(df.smile))
    pool.close()
    X=[np.array(fp).reshape(1, -1) for fp in fps]
    
    X = np.concatenate(X, axis=0)
    
    print('>>> computing predictions')
    y_score = model.predict_proba(X)[:,1]
    
    df['qsar']=y_score
    df.to_csv('excape_docked_lgbm.csv') # save scores to csv 
    
    # scatterplot
    dfa = df[df['active']=='A']
    plt.figure()
    plt.scatter(x=dfa['score'], y=dfa['qsar'])
    plt.ylim(0,1)
    plt.title('Correlation for experimental actives')
    plt.xlabel('drd3 docking')
    plt.ylabel('qsar score')
    
    dfi = df[df['active']=='N']
    plt.figure()
    plt.scatter(x=dfi['score'], y=dfi['qsar'])
    plt.ylim(0,1)
    plt.title('Correlation for experimental inactives')
    plt.xlabel('drd3 docking')
    plt.ylabel('qsar score')

    positives = df[df['qsar']>=0.5]
    print(positives.shape[0], f' positives')
    
    


