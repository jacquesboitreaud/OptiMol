# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:27:36 2020

@author: jacqu

Use QSAR to score the CbAS samples 

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

import os, sys
from multiprocessing import Pool

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '../..'))

def process_one(s):
    m = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
    return fp

if __name__=='__main__':
    
    
    
    with open('../../results/saved_models/qsar_svm.pickle', 'rb') as f :
        clf = pickle.load(f)
    print('Loaded svm ')
    
    
    df = pd.read_csv('violin_data.csv')
    print(f'>>> computing fps for samples')

    X=[]
    pool=Pool()
    fps = pool.map(process_one, list(df.smile))
    pool.close()
    X=[np.array(fp).reshape(1, -1) for fp in fps]
    
    X = np.concatenate(X, axis=0)
    
    print('>>> computing predictions')
    y_score = clf.predict_proba(X)[:,1]
    
    df['qsar']=y_score
    df.to_csv('violin_data_qsar.csv')
    
    # scatterplot
    plt.figure()
    plt.scatter(x=df['score'], y=df['qsar'])
    plt.ylim(0,1)
    plt.xlabel('drd3 docking')
    plt.ylabel('qsar score')

    positives = df[df['qsar']>=0.5]
    print(positives.shape[0], f' positives')
    
    


