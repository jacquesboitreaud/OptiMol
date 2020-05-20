# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

CHecks presence of actives in samples 

"""
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import argparse

import selfies 
from ordered_set import OrderedSet

if __name__=='__main__':

    with open('../data/gen.txt', 'r') as f :
        smiles = f.readlines()
        
    actives_set = pd.read_csv('../data/finetuning.csv')
    actives_set = set(list(actives_set.smiles))
    cpt=0 
    discovered = []
    
    kek_actives = []
    
    for s in actives_set :
        m=Chem.MolFromSmiles(s)
        Chem.Kekulize(m)
        s2 = Chem.MolToSmiles(m, kekuleSmiles=True)
        kek_actives.append(s2)
        
    actives_set = set(kek_actives)
    
    for s in smiles :
        m=Chem.MolFromSmiles(s.rstrip())
        Chem.Kekulize(m)
        s = Chem.MolToSmiles(m, kekuleSmiles=True)
        
        if s in actives_set :
            cpt +=1
            discovered.append(s)
            print('active')
            
    d=np.unique(discovered)
    
    print(f'sampled {d.shape[0]} actives from the train set')
            
    
