# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:36:51 2019

@author: jacqu

Go over molecules dataset and collects edge types for one-hot encoding

Stores dict 'order2index' to pickle file
"""
import pandas as pd 
import pysmiles
import networkx as nx
import numpy as np
from tqdm import tqdm
import pickle
from multiprocessing import Pool, freeze_support, RLock

from rdkit_to_nx import smiles_to_nx

def main():
    freeze_support()
    # PARAMS : path to data and suffix for created pickle file. 
    suffix = ''
    data = pd.read_csv('../data/moses_train.csv', nrows=100)
    
    #=========================================================================
    smiles = list(data['can'])
    # EDGE attributes
    edge_types = set()
    # ATOM types 
    chiral_types = set()
    atom_types = set()
    formal_charges = set()
    
    print("Collecting edge data...")
    
    
    for s in tqdm(smiles):
        graph = smiles_to_nx(s)
        for _,_,e_dict in graph.edges(data=True):
            if(e_dict['bond_type'] not in edge_types):
                edge_types.add(e_dict['bond_type'])
                
        for n, n_dict in graph.nodes(data=True):
            if(n_dict['chiral_tag'] not in chiral_types):
                chiral_types.add(n_dict['chiral_tag'])
            if(n_dict['atomic_num'] not in atom_types):
                atom_types.add(n_dict['atomic_num'])
            if(n_dict['formal_charge'] not in formal_charges):
                formal_charges.add(n_dict['formal_charge'])
              
    print('Edge types found: \n',edge_types)
    print('Charges found: \n', formal_charges)
    print('Atom types found: \n',atom_types)
    print('Chiral types found: \n', chiral_types)
    
    # Turn sets to dict and save : 
    edge_types=list(edge_types)
    edge_types = {t:i for (i,t) in enumerate(edge_types)}
    
    chiral_types = list(chiral_types)
    chiral_types = {t:i for (i,t) in enumerate(chiral_types)}
    atom_types = list(atom_types)
    atom_types = {t:i for (i,t) in enumerate(atom_types)}
    formal_charges = list(formal_charges)
    formal_charges= {t:i for (i,t) in enumerate(formal_charges)}
    
    #np.save('../edge_map.npy',edge_map)
    
    with open(f"edges_and_nodes_map{suffix}.pickle","wb") as f:
        pickle.dump(edge_types, f)
        pickle.dump(atom_types, f)
        pickle.dump(chiral_types, f)
        pickle.dump(formal_charges, f)
    
if(__name__=='__main__'):
    main()