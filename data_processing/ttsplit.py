# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:14:43 2020

@author: jacqu

Consistent test-train split for DUDE and CHEMBL 
"""
import pandas as pd 
import numpy as np
import os


dud = pd.read_csv('../data/DUD_clean.csv')

# 1 - 80/20 test/train split

targets = os.listdir('C:/Users/jacqu/Documents/mol2_resource/dud/all')

cpt=0

for t in targets : 
    cpt+=1
    dft = dud[dud[t]!=0]
    
    if(cpt==1):
        train=dft.sample(frac=0.8,random_state=200) #random state is a seed value
        test=dft.drop(train.index)
    else:
        train_app=dft.sample(frac=0.8,random_state=200) #random state is a seed value
        test_app=dft.drop(train_app.index)
        
        train=pd.concat([train,train_app])
        test=pd.concat([test,test_app])
        
train.to_csv('../data/train.csv')
test.to_csv('../data/test.csv')

# 2 - Select only two targets for the finetuning step 

gpcr = ['aa2ar','adrb1','adrb2','cxcr4','drd3']

# select only these in whole DUDE, and do 80% / 20% split 
cpt=0
for g in gpcr : 
    cpt+=1
    dft = dud[dud[g]!=0]
    
    if(cpt==1):
        train=dft.sample(frac=0.8,random_state=200) #random state is a seed value
        test=dft.drop(train.index)
    else:
        train_app=dft.sample(frac=0.8,random_state=200) #random state is a seed value
        test_app=dft.drop(train_app.index)
        
        train=pd.concat([train,train_app])
        test=pd.concat([test,test_app])
        

        
train.to_csv('../data/train_gpcr.csv')
test.to_csv('../data/test_gpcr.csv')  
        
# Clean columns for train_app and test_app 
        
        
        
# Build triplets







