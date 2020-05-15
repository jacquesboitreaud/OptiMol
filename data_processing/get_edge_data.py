# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:36:51 2019

@author: jacqu

Go over molecules dataset and collects edge types for one-hot encoding

Stores dict 'order2index' to pickle file
"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import argparse
from multiprocessing import Pool, freeze_support, RLock
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

from data_processing.rdkit_to_nx import smiles_to_nx

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-csv', '--csv', help="path to molecules dataset. Column with smiles labeled 'can'", type=str,
                        default='../data/moses_train.csv')

    # =======

    args, _ = parser.parse_known_args()

    data = pd.read_csv(args.csv)
    try:
        smiles = list(data['can'])
    except:
        print('No column with label "can" found in csv file. Assuming canonical smiles are in column 0.')
        smiles = list(data.iloc[0])

    print('Parsing ', len(smiles), ' molecules')

    # =========================================================================

    # EDGE attributes
    edge_types = set()
    # ATOM types 
    chiral_types = set()
    atom_types = set()
    formal_charges = set()

    print("Collecting edge data...")

    for s in tqdm(smiles):
        graph = smiles_to_nx(s)
        for _, _, e_dict in graph.edges(data=True):
            if (e_dict['bond_type'] not in edge_types):
                edge_types.add(e_dict['bond_type'])

        for n, n_dict in graph.nodes(data=True):
            if (n_dict['chiral_tag'] not in chiral_types):
                chiral_types.add(n_dict['chiral_tag'])
            if (n_dict['atomic_num'] not in atom_types):
                atom_types.add(n_dict['atomic_num'])
            if (n_dict['formal_charge'] not in formal_charges):
                formal_charges.add(n_dict['formal_charge'])

    print('Edge types found: \n', edge_types)
    print('Charges found: \n', formal_charges)
    print('Atom types found: \n', atom_types)
    print('Chiral types found: \n', chiral_types)

    # Turn sets to dict and save : 
    edge_types = list(edge_types)
    edge_types = {t: i for (i, t) in enumerate(edge_types)}

    chiral_types = list(chiral_types)
    chiral_types = {t: i for (i, t) in enumerate(chiral_types)}
    atom_types = list(atom_types)
    atom_types = {t: i for (i, t) in enumerate(atom_types)}
    formal_charges = list(formal_charges)
    formal_charges = {t: i for (i, t) in enumerate(formal_charges)}

    with open("map_files/edges_and_nodes_map.pickle", "wb") as f:
        pickle.dump(edge_types, f)
        pickle.dump(atom_types, f)
        pickle.dump(chiral_types, f)
        pickle.dump(formal_charges, f)
    print('Successfully built features maps. ')
