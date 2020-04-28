# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Compute selfies for all smiles in csv 
"""
import pandas as pd 
import argparse
from tqdm import tqdm

try: 
    from selfies import encoder, decoder, selfies_alphabet
except ImportError:
    print('Please install selfies package by running "pip install selfies" ')
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--smiles_file', help="path to csv with dataset", type=str, default='data/moses_train.csv')
    
    # ======================
    args=parser.parse_args()
    
    print(f'>>> Computing selfies for all smiles in {args.smiles_file}. May take some time.')
    train = pd.read_csv(args.smiles_file, index_col = 0)
    
    selfies_list=[]
    max_selfies_len = 0
    max_smiles_len = 0 
    
    for s in tqdm(train.smiles):
        
        if(len(s)>max_smiles_len):
            max_smiles_len = len(s)
        individual_selfie = encoder(s)
        s_len = len(individual_selfie)-len(individual_selfie.replace('[',''))
        if s_len > max_selfies_len:
            max_selfies_len = s_len 
            
        selfies_list.append(encoder(s))
        
    train['selfies']=pd.Series(selfies_list, index=train.index)
    
    train.to_csv(args.smiles_file)
    print('Saved selfies as a column in csv. Ready to train with selfies.')
    print(f'Max smiles length : {max_smiles_len}')
    print(f'Max selfies length : {max_selfies_len}')