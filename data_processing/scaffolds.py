# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

Finds and stores unique scaffolds in dataset. 
Scaffolds are Murcko scaffolds, computed using RDKIT module MurckoScaffold.MurckoScaffoldSmiles. 

"""
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-csv', '--csv', help="path to molecules dataset. Column with smiles labeled 'can'", type=str, default='../data/moses_train.csv')
    parser.add_argument('-n', '--cutoff', help="cutoff N molecules. -1 for all in csv", type=int, default=10)
    # =======

    args=parser.parse_args()
    
    if(args.cutoff>0):
        data = pd.read_csv(args.csv, nrows = args.cutoff)
    else:
        data = pd.read_csv(args.csv)
        
    try:
        smiles = list(data['can'])
    except:
        print('No column with label "can" found in csv file. Assuming canonical smiles are in column 0.')
        smiles = list(data.iloc[0])

    print('>>> Parsing ', len(smiles), ' molecules')

    

    scaffolds_dict = {}
    scaffolds = []
    
    
    for i,s in enumerate(smiles) : 
        if (i%1000==0 and i>0):
            print(i)
        sc = MurckoScaffold.MurckoScaffoldSmiles(s)
        scaffolds.append(sc)
        if(sc not in scaffolds_dict.keys()):
            scaffolds_dict[sc]=1
        else:
            scaffolds_dict[sc]+=1
    
    scaf, counts = [k for k,v in scaffolds_dict.items()], [v for k,v in scaffolds_dict.items()]
    print(f'Found {len(scaf)} unique scaffolds in dataset')
    output_d = {'smiles':scaf, 'counts':counts}
        
    # Scaffolds and counts 
    u = pd.DataFrame.from_dict(output_d)
    print(f'>>> Saving scaffolds and number of occurences to ~/unique_scaffolds.csv')
    u.to_csv('unique_scaffolds.csv')
        
    # For each molecule, add its scaffold as a df column
    data['scaffold']= pd.Series(scaffolds, index = data.index)
    
    savename = args.csv[:-4]+'_scaf.csv'
    print(f'>>> saving dataframe to {savename}')
    data.to_csv(savename)
    
