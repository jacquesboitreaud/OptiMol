# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:53:42 2019

@author: jacqu

Logistic regression using learned latent rpz for affinity prediction 

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

from rdkit import Chem
from rdkit.Chem import AllChem

import pybel

import sys
if(__name__=='__main__'):
    sys.path.append('eval')

from model_utils import load_trained_model, embed

def evaluate_LR(features, labels):
    # 3-fold fitting and scoring of logistic regression 
    X,y = features, labels
    cv = KFold(n_splits = 3, shuffle = True)
    clf = LogisticRegression()
    avg_lr_f1 = 0
    i=0
    for train_index, test_index in cv.split(X):
        i+=1
        print('============= Training and evaluating on split n°{} ================'.format(i))
        # Split
        X_train= [X[k] for k in train_index]
        X_test = [X[k] for k in test_index]
        y_train = [y[k] for k in train_index]
        y_test = [y[k] for k in test_index]
        
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        
        score = f1_score(y_test, y_pred )
        print(f'Split n°{i}, accuracy : {score}')
        avg_lr_f1 += score
        
    avg_lr_f1 = avg_lr_f1/(i)
    print(f'kfold lr F1 score : {avg_lr_f1}')
    
    return clf

def morgan_fp(df):
    # computes morgan fingerprints for molecules in df 
    n = df.shape[0]
    X = np.zeros((n,166)) # MACCS fingerprints are dim 166
    smiles = list(df['can'])[:4]
    for i,s in enumerate(smiles):
        m= pybel.readstring("smi", s)
        fp = m.calcfp("ecfp4", 1024).bits 
        print(fp)
        X[i,fp]=1
    return X 
    
# ==================================================================================
# 1 / Load the DUD ligands and decoys dataframe 
# ==================================================================================

#df=pd.read_csv('data/dude_targets/esr1.csv', nrows = 10000)
#target = 'esr1'
#labels = np.array(df[target])

# 2 / Compute baseline embeddings (ECFP, molecularVAE embedding... )

# TODO
# z_ecfp = 


# TODO 
#z_vae = 


# Affinity shaped vae 
z_affvae = np.concatenate((z_a,z_a2))
na, nn = z_a.shape[0], z_a2.shape[0]
labels = np.concatenate((np.ones(na), np.zeros(nn)))

clf = evaluate_LR(z_affvae, labels)




    
    