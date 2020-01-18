# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for passing triplets of molecules to model

"""

import os 
import sys
if __name__ == "__main__":
    sys.path.append("..")
    sys.path.append("data_processing")
    
import torch
import dgl
import pandas as pd
import numpy as np

import pickle
import json
import networkx as nx
from torch.utils.data import Dataset, DataLoader, Subset
from rdkit_to_nx import smiles_to_nx


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, label).
    data_i, data_j, data_l = map(list,zip(*samples))
    
    g_i, s_i, p_i, a_i = map(list, zip(*data_i))
    g_j, s_j, p_j, a_j = map(list, zip(*data_j))
    g_l, s_l, p_l, a_l = map(list, zip(*data_l))
    
    bgi, bgj, bgl = dgl.batch(g_i), dgl.batch(g_j), dgl.batch(g_l)
    
    s_i, s_j, s_l = [torch.tensor(s, dtype = torch.long) for s in [s_i,s_j,s_l]]
    p_i,p_j,p_l = [torch.tensor(p_labels) for p_labels in [p_i,p_j,p_l]]
    a_i,a_j,a_l = [torch.tensor(a_labels) for a_labels in [a_i,a_j,a_l]]
    
    
    return bgi,s_i,p_i,a_i, bgj, s_j, p_j, a_j, bgl, s_l, p_l, a_l

def oh_tensor(category, n):
    t = torch.zeros(n,dtype=torch.float)
    t[category]=1
    return t


class molDataset(Dataset):
    """ 
    pytorch Dataset for training on small molecules graphs + smiles 
    """
    def __init__(self, csv_path,
                n_mols,
                props, 
                targets,
                debug=False):
        
        if(n_mols!=-1):
            self.df = pd.read_csv(csv_path, nrows=n_mols)
            self.n = n_mols
            print('columns:', self.df.columns)
        else:
            self.df = pd.read_csv(csv_path)
            self.n = self.df.shape[0]
            print('columns:', self.df.columns)
        
        # 1/ ============== Properties & Targets handling: ================
        
        self.targets = targets
        self.props = props
        print(f'Labels retrieved for the following {len(self.targets)} targets: {self.targets}')
        
        # =========== 2/ Graphs handling ====================
        
        mapspath='map_files/edges_and_nodes_map.pickle'
        
        with open(mapspath,"rb") as f:
            self.edge_map= pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map= pickle.load(f)
            self.charges_map = pickle.load(f)
            
        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('Loaded edge and atoms types maps.')
        
        self.emb_size = self.num_atom_types + self.num_charges # node embedding size
        
        # 3/ =========== SMILES handling : ==================
        
        char_file="map_files/zinc_chars.json"
        self.char_list = json.load(open(char_file))
        self.char_to_index= dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char= dict((i, c) for i, c in enumerate(self.char_list))
        self.n_chars=len(self.char_list)
        
        self.max_smi_len = 151 # can be changed 
        print(f"Loaded smiles characters file. Max smiles length is {self.max_smi_len}")
        
        if(debug):
            # special case for debugging
            pass
            
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        # Return as dgl graph, smiles (indexes), targets properties.
        
        row = self.df.iloc[idx]
        smiles, props, targets = row['can'], \
        np.array(row[self.props],dtype=np.float32), np.array(row[self.targets],dtype=np.float32)
        
        
        # Checks
        if(len(smiles)>self.max_smi_len):
            print(f'smiles length error: l={len(smiles)}, longer than {self.max_smi_len}')
        
        targets = np.nan_to_num(targets)
        
        # 1 - Graph building
        graph=smiles_to_nx(smiles)
        
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'bond_type')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)
        
        try:
            at_type = {a: oh_tensor(self.at_map[label], self.num_atom_types) for a, label in
                       (nx.get_node_attributes(graph, 'atomic_num')).items()}
            nx.set_node_attributes(graph, name='atomic_num', values=at_type)
        except KeyError:
            print(smiles)
        
        at_charge = {a: oh_tensor(self.charges_map[label], self.num_charges) for a, label in
                   (nx.get_node_attributes(graph, 'formal_charge')).items()}
        nx.set_node_attributes(graph, name='formal_charge', values=at_charge)
        
        
        at_chir = {a: torch.tensor(self.chi_map[label]) for a, label in
                   (nx.get_node_attributes(graph, 'chiral_tag')).items()}
        nx.set_node_attributes(graph, name='chiral_tag', values=at_chir)
        
        # to dgl 
        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, 
                            node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic'],
                            edge_attrs=['one_hot'])

        g_dgl.ndata['h'] =torch.cat([g_dgl.ndata['formal_charge'], g_dgl.ndata['atomic_num']], dim=1)
        
        # 2 - Smiles array 
        a = np.zeros(self.max_smi_len)
        idces = [self.char_to_index[c] for c in smiles]
        idces.append(self.char_to_index['\n'])
        a[:len(idces)]=idces
        
        return g_dgl, a, props, targets
        
    
class tripletDataset(Dataset):
    # Object that yields triplets of molecules coming from two datasets 
    def __init__(self, actives_D, decoys_D):
        # Initialize from ligands dataset and decoys_dataset
        self.na, self.nd = len(actives_D), len(decoys_D)
        
        self.actives_D = actives_D
        self.decoys_D = decoys_D
        
        
    def __len__(self):
        return min((self.na**2)*self.nd,1000000) # max 1M samples 
        
    def __getitem__(self,idx):
        # Select random indices for triplet
        a = np.random.randint(0,self.na,2)
        d = np.random.randint(0,self.nd)
        
        # Get the corresponding molecules from their dataset
        g_i, a_i, props_i, targets_i = self.actives_D.__getitem__(a[0])
        g_j, a_j, props_j, targets_j = self.actives_D.__getitem__(a[1])
        g_l, a_l, props_l, targets_l = self.decoys_D.__getitem__(d)
        
        # Assemble the 3 
        
        return [g_i,a_i,props_i, targets_i], [g_j, a_j, props_j, targets_j], [g_l, a_l, props_l, targets_l]
        
    
class Loader():
    def __init__(self,
                 actives_csv,
                 decoys_csv,
                 props,
                 targets,
                 batch_size=64,
                 num_workers=20,
                 debug=False,
                 test_only=False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        
        if test_only: puts all molecules in csv in the test loader. Returns empty train and valid loaders
        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.actives_dataset = molDataset(actives_csv,
                                          n_mols=-1,
                                  debug=debug,
                                  props = props,
                                  targets=targets)
        self.decoys_dataset = molDataset(decoys_csv,
                                         n_mols=-1,
                                          debug=debug,
                                          props = props,
                                          targets=targets)
        
        self.t_dataset = tripletDataset(self.actives_dataset,
                                        self.decoys_dataset)
                        
        # Both datasets should have same maps 
        self.num_edge_types, self.num_atom_types = self.actives_dataset.num_edge_types, self.actives_dataset.num_atom_types
        self.num_charges= self.actives_dataset.num_charges
        self.test_only=test_only
        
    def get_maps(self):
        # Returns dataset mapping of edge and node features 
        return self.actives_dataset.edge_map, self.actives_dataset.at_map, self.actives_dataset.chi_map, self.actives_dataset.charges_map
    
    def get_reverse_maps(self):
        # Returns maps of one-hot index to actual feature 
        rev_em = {v:i for (i,v) in self.actives_dataset.edge_map.items() }
        rev_am = {v:i for (i,v) in self.actives_dataset.at_map.items() }
        rev_chi_m={v:i for (i,v) in self.actives_dataset.chi_map.items() }
        rev_cm={v:i for (i,v) in self.actives_dataset.charges_map.items() }
        return rev_em, rev_am, rev_chi_m, rev_cm

    def get_data(self):
        na = len(self.actives_dataset)
        nd = len(self.decoys_dataset)
        print(f"Splitting datasets with {na} actives and {nd} decoys")
        
        if(not self.test_only):
            train_loader = DataLoader(dataset=self.t_dataset, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)

        return train_loader
    
if(__name__=='__main__'):
    d=molDataset()