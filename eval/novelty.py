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

from time import time

from rdkit import Chem
from rdkit.Chem import Draw 

import matplotlib.pyplot as plt

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "--generated_samples", help="samples", type=str, default='data/gen.txt')
    parser.add_argument('-t', "--training_samples", help="training csv", type=str, default='shuffled_whole_zinc.csv')
    parser.add_argument('-N', "--cutoff", help="Cutoff rows for training csv , if very large", type=int, default=22400000)

    args, _ = parser.parse_known_args()
    # =======================================
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, 'dataloaders'))
    sys.path.append(os.path.join(script_dir, 'data_processing'))

    with open(os.path.join(script_dir,'..',args.generated_samples), 'r') as f :
        smiles_list = [line.rstrip() for line in f]
        
    training_set = pd.read_csv(os.path.join(script_dir,'..','data', args.training_samples), nrows = args.cutoff)
    training_set = set(list(training_set.smiles))
    
    novel = 0 
    for s in smiles_list :
        
        m=Chem.MolFromSmiles(s)
        if m is not None :
            #s=Chem.MolToSmiles(m, kekuleSmiles=True)
            if s not in training_set :
                novel +=1
            
    print(novel/len(smiles_list))
    
    ## Diversity sampling
    
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
    from rdkit import DataStructs
    from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
    
    ms = [Chem.MolFromSmiles(s) for s in smiles_list]
    start = time()
    fps = [GetMorganFingerprint(x,3) for x in ms]
    nfps = len(fps)
    end = time()
    print(f'Time for {nfps} fingerprints: ', end-start)
    
    
    def distij(i,j,fps=fps):
        return 1-DataStructs.DiceSimilarity(fps[i],fps[j])
    
    picker = MaxMinPicker()
    start = time()
    pickIndices = picker.LazyPick(distij,nfps,1000,seed=23)
    end = time()
    idces = list(pickIndices)
    print('Time for picker: ', end-start)
    

    m_selected = [ms[i] for i in idces]
    img = Draw.MolsToGridImage(m_selected)
    img
    img.save('diverse.png')