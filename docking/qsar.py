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
import pandas as pd 

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

df = pd.read_csv('../data/excape_drd3.csv')

smiles = df.smile
activities = df.active
n_bits = 2048


X, Y = [], []

for i,(s, a) in enumerate(zip(smiles, activities)):
    m = Chem.MolFromSmiles(s)
    
    if m!=None :
        fp = AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=n_bits) # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
    
        X.append(np.array(fp).reshape(1,-1))
        if a =='A':
            Y.append(1)
        else:
            Y.append(0)
        
X = np.concatenate(X, axis = 0)
Y = np.array(Y)

print('Number of samples :', Y.shape[0])

# Train test split 

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.5, random_state=42)

# Fitting SVM 

clf = SVC(gamma='auto')
print('>>> Fitting SVM to train molecules')
clf.fit(X_train, y_train)

sc = clf.score(X_test, y_test)
print(' Test set accuracy : ', sc)
