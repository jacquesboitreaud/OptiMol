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
from sklearn.metrics import f1_score, accuracy_score

from rdkit import Chem
from rdkit.Chem import AllChem

import pybel

import sys
if(__name__=='__main__'):
    sys.path.append('eval')


def evaluate_LR(features, labels, exp_id):
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
        
        score = accuracy_score(y_test, y_pred )
        #print(f'Split n°{i}, accuracy : {score}')
        avg_lr_f1 += score
        
    avg_lr_f1 = avg_lr_f1/(i)
    print(f'kfold lr F1 score for {exp_id} : {avg_lr_f1}')
    
    return clf, avg_lr_f1

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

tars = ['herg actives', 'drd3 actives', 'common actives']

f1_r = []

# Affinity shaped vae 
for i in range(len(Z)):
    for j in  range(len(Z)):
        if(i!=j):
            pos = Z[i][:227]
            neg = Z[j][:227]
            z_affvae = np.concatenate((pos,neg))
            na, nn = pos.shape[0], neg.shape[0]
            labels = np.concatenate((np.ones(na), np.zeros(nn)))
            
            clf, sc = evaluate_LR(z_affvae, labels, exp_id=f'{tars[i]} - {tars[j]}')
            
            f1_r.append(sc)

ar=np.arange(len(f1_r))*3
plt.bar(ar, f1_b)
plt.bar(ar+1, f1_r)
plt.show()
    
    