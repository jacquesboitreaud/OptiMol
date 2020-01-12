# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

COmpute chemical properties for a dataframe 
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import pandas as pd

#######" CHoose local or server 
LOCAL = False

if(LOCAL):
    DUD = pd.read_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL_18t.csv')
else:
    DUD = pd.read_csv('/home/mcb/jboitr/data/CHEMBL_18t.csv')
    
    
smiles = list(DUD['can'])
d={}
prop_names=['QED','logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD','jIndex']
for name in prop_names : 
    d[f'{name}']=[]


for i,s in enumerate(smiles) : 
    if(i%10000==0):
        print(i)
    m=Chem.MolFromSmiles(s)
    if(m==None or 'i' in s or '.' in s):
        DUD=DUD.drop(i)
        print(s, i)
    else:
        d['QED'].append(QED.default(m))
        d['logP'].append(Crippen.MolLogP(m))
        d['molWt'].append(Descriptors.MolWt(m))
        d['maxCharge'].append(Descriptors.MaxPartialCharge(m))
        d['minCharge'].append(Descriptors.MinPartialCharge(m))
        d['valence'].append(Descriptors.NumValenceElectrons(m))
        d['TPSA'].append(rdMolDescriptors.CalcTPSA(m))
        d['HBA'].append(rdMolDescriptors.CalcNumHBA(m))
        d['HBD'].append(rdMolDescriptors.CalcNumHBD(m))
        d['jIndex'].append(GraphDescriptors.BalabanJ(m))
    

for k in d.keys():
    DUD[k]=pd.Series(d[k], index = DUD.index)
    
# Drop lines with Nan properties
DUD = DUD.dropna(axis=0)

if(LOCAL):
    DUD.to_csv('C:/Users/jacqu/Documents/GitHub/data/CHEMBL_18t_p.csv')
else:
    DUD.to_csv('/home/mcb/jboitr/data/CHEMBL_18t.csv')
    
