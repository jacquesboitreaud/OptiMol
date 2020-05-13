# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

Compute chemical properties for a dataframe
"""
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import pandas as pd

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', '--csv', help="path to molecules dataset. Column with smiles labeled 'can'", type=str, default='../data/moses_train.csv')
    parser.add_argument('-n', '--cutoff', help="cutoff N molecules. -1 for all in csv", type=int, default=10)
    
    # =======

    args, _ = parser.parse_known_args()
    
    if(args.cutoff>0):
        data = pd.read_csv(args.csv, nrows = args.cutoff)
    else:
        data = pd.read_csv(args.csv)
    
    smiles = list(data['can'])
    d={}
    prop_names=['QED','logP','molWt']
    for name in prop_names : 
        d[f'{name}']=[]
    
    print(f'>>> computing {prop_names} for {len(smiles)} molecules')
    for i,s in enumerate(smiles) : 
        if(i%1000==0 and i >0):
            print(i)
        m=Chem.MolFromSmiles(s)
        d['QED'].append(QED.default(m))
        d['logP'].append(Crippen.MolLogP(m))
        d['molWt'].append(Descriptors.MolWt(m))
        
    
    for k in d.keys():
        data[k]=pd.Series(d[k], index = data.index)
        
    # Drop lines with Nan properties
    data = data.dropna(axis=0, subset = prop_names)
    
    savename = args.csv[:-4]+'_props.csv'
    print(f'>>> saving dataframe to {savename}')
    data.to_csv(savename)
    
