# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for SMILES to graph 

"""

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
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
    graphs, smiles, p_labels, a_labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    p_labels, a_labels = torch.tensor(p_labels), torch.tensor(a_labels)
    smiles = torch.tensor(smiles, dtype=torch.long)

    return batched_graph, smiles, p_labels, a_labels


def oh_tensor(category, n):
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1
    return t


class molDataset(Dataset):
    """ 
    pytorch Dataset for training on small molecules graphs + smiles 
    """

    def __init__(self, csv_path,
                 maps_path,
                 n_mols=-1,
                 props=None,
                 targets=None,
                 debug=False,
                 shuffle=False,
                 select_target=None):

        print('**************** dataset building ******************')
        # 1/ two solutions: dataframe given or csv path given 
        if (csv_path is None):
            print("Empty dataset initialized. Use pass_dataset or pass_dataset_path to add molecules.")
            self.df = None
            self.n = 0

        else:
            # Cutoff number of molecules 
            if (n_mols != -1):
                self.df = pd.read_csv(csv_path, nrows=n_mols)
                self.n = n_mols
                print('columns:', self.df.columns)
            else:
                self.df = pd.read_csv(csv_path)
                self.n = self.df.shape[0]
                print('Dataset columns:', self.df.columns)

            if (select_target != None):  # keep only actives and decoys for a specific target (use for test plots)
                print('Restricting dataset to a selected target')
                self.df = self.df[self.df[select_target] != 0]
                self.n = self.df.shape[0]
                print('Dataset columns:', self.df.columns)

        # 1/ ============== Properties & Targets handling: ================

        self.targets = targets
        self.props = props

        # =========== 2/ Graphs handling ====================

        with open(os.path.join(maps_path, 'edges_and_nodes_map.pickle'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('Loaded edge and atoms types maps.')

        self.emb_size = self.num_atom_types + self.num_charges  # node embedding size

        # 3/ =========== SMILES handling : ==================

        char_file = os.path.join(maps_path, "zinc_chars.json")
        self.char_list = json.load(open(char_file))
        self.char_to_index = dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.char_list))
        self.n_chars = len(self.char_list)

        self.max_smi_len = 151  # can be changed
        print(f"Loaded smiles characters file. Max smiles length is {self.max_smi_len}")

        if (debug):
            # special case for debugging
            pass

    def pass_dataset_path(self, path):
        # Pass a new dataset to the loader, without changing other parameters 
        self.df = pd.read_csv(path)
        self.n = self.df.shape[0]
        print('New dataset columns:', self.df.columns)

    def pass_dataset(self, df):
        self.df = df
        self.n = df.shape[0]
        print('New dataset columns:', self.df.columns)

    def pass_smiles_list(self, smiles):
        # pass smiles list to the model; a dataframe with unique column 'can' will be created 
        self.df = pd.DataFrame.from_dict({'can': smiles})
        self.n = self.df.shape[0]
        print('New dataset contains only can smiles // no props or affinities')

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Return as dgl graph, smiles (indexes), targets properties.

        row = self.df.iloc[idx]
        smiles = row['can']
        # Checks
        if (len(smiles) > self.max_smi_len):
            print(f'smiles length error: l={len(smiles)}, longer than {self.max_smi_len}')

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

        # 3 - Optional props and affinities 
        props, targets = 0, 0
        if (self.props != None):
            props = np.array(row[self.props], dtype=np.float32)

        if (self.targets != None):
            try:
                targets = np.array(row[self.targets], dtype=np.float32)
            except:
                targets = np.zeros(len(self.targets), dtype=np.float32)
            targets = np.nan_to_num(targets)

        return g_dgl, a, props, targets


class Loader():
    def __init__(self,
                 csv_path=None,
                 maps_path ='../map_files/',
                 n_mols=None,
                 props=None,
                 targets=None,
                 batch_size=64,
                 num_workers=20,
                 debug=False,
                 test_only=False,
                 shuffle=False,
                 select_target=None):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        if test_only: puts all molecules in csv in the test loader. Returns empty train and valid loaders

        """

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = molDataset(csv_path,
                                  maps_path, 
                                  n_mols,
                                  debug=debug,
                                  props=props,
                                  targets=targets,
                                  shuffle=shuffle,
                                  select_target=select_target)

        self.num_edge_types, self.num_atom_types = self.dataset.num_edge_types, self.dataset.num_atom_types
        self.num_charges = self.dataset.num_charges
        self.test_only = test_only

    def get_maps(self):
        # Returns dataset mapping of edge and node features 
        return self.dataset.edge_map, self.dataset.at_map, self.dataset.chi_map, self.dataset.charges_map

    def get_reverse_maps(self):
        # Returns maps of one-hot index to actual feature 
        rev_em = {v: i for (i, v) in self.dataset.edge_map.items()}
        rev_am = {v: i for (i, v) in self.dataset.at_map.items()}
        rev_chi_m = {v: i for (i, v) in self.dataset.chi_map.items()}
        rev_cm = {v: i for (i, v) in self.dataset.charges_map.items()}
        return rev_em, rev_am, rev_chi_m, rev_cm

    def get_data(self):
        n = len(self.dataset)
        
        indices = list(range(n))
        np.random.shuffle(indices)
        if (not self.test_only):
            split_train, split_valid = 0.95, 0.95
            train_index, valid_index = int(split_train * n), int(split_valid * n)

        else:
            split_train, split_valid = 0, 0
            train_index, valid_index = 0, 0

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)
        print(f"Dataset contains {n} samples (train subset: {len(train_set)}, Test subset:{len(test_set)}) ")
        print(f"Train subset contains {len(train_set)} samples")
        print(f"Test subset contains {len(test_set)} samples")

        if (not self.test_only):
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block, drop_last=True)

        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)

        test_loader = DataLoader(dataset=test_set, shuffle=not self.test_only, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block, drop_last=True)

        # return train_loader, valid_loader, test_loader
        if (not self.test_only):
            return train_loader, 0, test_loader
        else:
            return 0, 0, test_loader


if __name__ == '__main__':
    L=Loader(csv_path=None)
