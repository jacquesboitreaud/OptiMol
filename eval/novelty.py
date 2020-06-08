# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:30:12 2020

@author: jacqu

Moses metrics to evaluate quality and diversity of samples 

"""
import moses 
import os 
import sys
import argparse
import pandas as pd

from rdkit import Chem

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--generated_samples", help="Nbr to generate", type=str, default='data/gen.txt')

    args, _ = parser.parse_known_args()
    # =======================================
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, 'dataloaders'))
    sys.path.append(os.path.join(script_dir, 'data_processing'))

    with open(os.path.join(script_dir,'..',args.generated_samples), 'r') as f :
        smiles_list = [line.rstrip() for line in f]
        
    training_set = pd.read_csv(os.path.join(script_dir,'..','data', '250k_zinc.csv'))
    training_set = set(list(training_set.smiles))
    
    novel = 0 
    for s in smiles_list :
        
        m=Chem.MolFromSmiles(s)
        if m is not None :
            s=Chem.MolToSmiles(m, kekuleSmiles=True)
            if s not in training_set :
                novel +=1
            
    print(novel/len(smiles_list))