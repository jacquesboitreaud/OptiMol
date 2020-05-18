# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for passing triplets of molecules to model

"""

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import torch
import dgl
import pandas as pd
import numpy as np

import pickle
import json
import networkx as nx
from torch.utils.data import Dataset, DataLoader, Subset

from data_processing.rdkit_to_nx import smiles_to_nx


def collate_block(samples):
    # Collates samples into a batch
    # The input `samples` is a list of pairs
    #  (graph, label).
    data_i, data_j, labels = map(list, zip(*samples))

    g_i, s_i, p_i, a_i = map(list, zip(*data_i))
    g_j, s_j, p_j, a_j = map(list, zip(*data_j))

    bgi, bgj = dgl.batch(g_i), dgl.batch(g_j)

    s_i, s_j = [torch.tensor(s, dtype=torch.long) for s in [s_i, s_j]]
    p_i, p_j = [torch.tensor(p_labels) for p_labels in [p_i, p_j]]
    a_i, a_j = [torch.tensor(a_labels) for a_labels in [a_i, a_j]]

    labels = torch.tensor(labels, dtype=torch.float)

    return bgi, s_i, p_i, a_i, bgj, s_j, p_j, a_j, labels


def oh_tensor(category, n):
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1
    return t


class molDataset(Dataset):
    """ 
    pytorch Dataset for training on small molecules graphs + smiles 
    """

    def __init__(self, csv_path,
                 n_mols,
                 props,
                 targets,
                 debug=False,
                 fold=None):

        if n_mols != -1:
            self.df = pd.read_csv(csv_path, nrows=n_mols)
            self.n = n_mols
            print('columns:', self.df.columns)
        else:
            self.df = pd.read_csv(csv_path)
            self.n = self.df.shape[0]
            print('columns:', self.df.columns)

        # Select only a part of the actives 
        if fold is not None:
            self.df = self.df[self.df['fold'] == fold]
            self.n = self.df.shape[0]
            self.df = self.df.reset_index(drop=True)

        # 1/ ============== Properties & Targets handling: ================

        self.targets = targets
        self.props = props

        # =========== 2/ Graphs handling ====================

        mapspath = 'map_files/edges_and_nodes_map.pickle'

        with open(mapspath, "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('Loaded edge and atoms types maps.')

        self.emb_size = self.num_atom_types + self.num_charges  # node embedding size

        # 3/ =========== SMILES handling : ==================

        char_file = "map_files/zinc_chars.json"
        self.char_list = json.load(open(char_file))
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))
        self.n_chars = len(self.char_list)

        self.max_smi_len = 151  # can be changed
        print(f"Loaded smiles characters file. Max smiles length is {self.max_smi_len}")

        if debug:
            # special case for debugging
            pass

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return as dgl graph, smiles (indexes), targets properties.

        row = self.df.iloc[idx]
        smiles = row['can']

        # 0 - Properties & Affinities or affinity profiles 
        props = np.array(row[self.props], dtype=np.float32)
        # TODO : Put back the true values if we use affinities labels
        targets = np.zeros(2, dtype=np.float32)
        # Binding profile for similarity training 
        profile = row['profile']
        # print(profile)

        # Checks
        if len(smiles) > self.max_smi_len:
            print(f'smiles length error: l={len(smiles)}, longer than {self.max_smi_len}')

        targets = np.nan_to_num(targets)

        # 1 - Graph building
        graph = smiles_to_nx(smiles)

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
                            node_attrs=['atomic_num', 'chiral_tag', 'formal_charge', 'num_explicit_hs', 'is_aromatic'],
                            edge_attrs=['one_hot'])

        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata['formal_charge'], g_dgl.ndata['atomic_num']], dim=1)

        # 2 - Smiles array 
        a = np.zeros(self.max_smi_len)
        idces = [self.char_to_index[c] for c in smiles]
        idces.append(self.char_to_index['\n'])
        a[:len(idces)] = idces

        return g_dgl, a, props, targets, profile


class pairDataset(Dataset):
    # Object that yields triplets of molecules coming from two datasets 

    def __init__(self, moldataset):

        self.df = moldataset
        self.n = moldataset.n

    def __len__(self):
        return 400000  # arbitrary set 1 epoch = 400k samples

    def __getitem__(self, idx):

        # Select random pairs. constraint ; first mol is active 
        i = np.random.randint(0, self.n)  # two random actives
        j = np.random.randint(0, self.n)  # two random actives

        g_i, a_i, props_i, targets_i, profile1 = self.df.__getitem__(i)
        g_j, a_j, props_j, targets_j, profile2 = self.df.__getitem__(j)

        # Adapt condition for positive pairs and negative pairs as desired 

        if profile1 == profile2:  # Both have same profile
            label = 1  # positive pair
        else:
            label = 0

        # Assemble the 2 + pair label 
        return [g_i, a_i, props_i, targets_i], [g_j, a_j, props_j, targets_j], label


class Loader():
    def __init__(self,
                 csv_data,
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
        if choose_fold is not none, training restricts to a fold of the data
        """
        self.choose_fold = 1

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.t1_actives = molDataset(csv_data,
                                     n_mols=-1,
                                     debug=debug,
                                     props=props,
                                     targets=targets,
                                     fold=self.choose_fold)
        """
        self.others = molDataset(clust2,
                                    n_mols=-1,
                                    debug=debug,
                                    props = props,
                                    targets=targets)
        """

        self.t_dataset = pairDataset(self.t1_actives)

        # Both datasets should have same maps 
        self.num_edge_types, self.num_atom_types = self.t1_actives.num_edge_types, self.t1_actives.num_atom_types
        self.num_charges = self.t1_actives.num_charges
        self.test_only = test_only

    def get_maps(self):
        # Returns dataset mapping of edge and node features 
        return self.t1_actives.edge_map, self.t1_actives.at_map, self.t1_actives.chi_map, self.t1_actives.charges_map

    def get_reverse_maps(self):
        # Returns maps of one-hot index to actual feature 
        rev_em = {v: i for (i, v) in self.t1_actives.edge_map.items()}
        rev_am = {v: i for (i, v) in self.t1_actives.at_map.items()}
        rev_chi_m = {v: i for (i, v) in self.t1_actives.chi_map.items()}
        rev_cm = {v: i for (i, v) in self.t1_actives.charges_map.items()}
        return rev_em, rev_am, rev_chi_m, rev_cm

    def get_data(self):
        if not self.test_only:
            train_loader = DataLoader(dataset=self.t_dataset, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)

        return train_loader


if __name__ == '__main__':
    d = molDataset()
