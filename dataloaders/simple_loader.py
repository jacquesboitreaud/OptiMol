# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

A loader that just passes graph and selfies for a plain g2s VAE 

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
from rdkit import Chem

import pickle
import json
import networkx as nx
from selfies import encoder, decoder
from ordered_set import OrderedSet

from torch.utils.data import Dataset, DataLoader
from data_processing.rdkit_to_nx import smiles_to_nx


def collate_block(samples):
    """
    Collates samples into batches.  
      removes 'None' graphs (reduces batch size)
    """

    samples = [s for s in samples if s[0]!=None]
    
    graphs, selfies,w = map(list, zip(*samples))

    batched_graph = dgl.batch(graphs)

    selfies = torch.tensor(selfies, dtype=torch.long)
    w = torch.tensor(w, dtype=torch.float)

    return batched_graph, selfies, w


def oh_tensor(category, n): 
    # One-hot float tensor construction
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1.0
    return t


class SimpleDataset(Dataset):
    """ 
    pytorch Dataset for training on small molecules graphs + smiles 
    """

    def __init__(self, 
                 maps_path,
                 vocab, debug = False):
        
        self.debug = debug # error prints for invalid smiles / mols

        
        # =========== 2/ Graphs handling ====================

        with open(os.path.join(maps_path, 'edges_and_nodes_map.pickle'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('> Loaded edge and atoms types to one-hot mappings')
        self.emb_size = 16  # node embedding size = number of node features 

        # 3/ =========== SMILES and SELFIES handling : ================== 
        
        self.language = vocab # smiles or selfies 

        print('-> Using PREDEFINED selfies alphabet // some strings may be ignored if not one-hot compatible')
        with open(os.path.join(maps_path, 'predefined_alphabets.pickle'), 'rb') as f :
            alphabets_dict = pickle.load(f)
            
        if(self.language == 'smiles'):
            self.alphabet = alphabets_dict['smiles_alphabet']
            self.max_len = alphabets_dict['largest_smiles_len']
            
        elif(self.language=='selfies'):
            self.alphabet = alphabets_dict['selfies_alphabet']
            self.max_len = alphabets_dict['largest_selfies_len']
            
        else:
            print("decode format not understood: 'smiles' or 'selfies' ")
            raise NotImplementedError
        
        self.char_to_index = dict((c, i) for i, c in enumerate(self.alphabet))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        self.n_chars = len(self.alphabet)

        print(f"> Loaded alphabet. Using {self.language}. Max sequence length allowed is {self.max_len}")

    def pass_smiles_list(self, smiles, weights):
        # pass smiles list to the model; a dataframe with unique column 'can' will be created 
        self.df = pd.DataFrame.from_dict({'smiles': smiles, 'weights':weights})
        self.n = self.df.shape[0]
        print('New dataset contains smiles and sample weights')
        self.input_type='smiles'
        
    def pass_selfies_list(self, selfies, weights):
        # pass smiles list to the model; a dataframe with unique column 'can' will be created 
        self.df = pd.DataFrame.from_dict({'selfies': selfies, 'weights':weights})
        self.n = self.df.shape[0]
        print('New dataset contains selfies and sample weights')
        self.input_type = 'selfies'

    def __len__(self):
        return self.n
    
    def _get_selfie_and_smiles_alphabets(self):
        """
        Returns alphabet and length of largest molecule in SMILES and SELFIES, for the training set passed.
            self.df Column's name must be 'smiles'.
        output:
            - selfies alphabet
            - longest selfies string
            - smiles alphabet (character based)
            - longest smiles string
        """
    
        smiles_list = np.asanyarray(self.df.smiles)
        
        selfies_list = np.asanyarray(self.df.selfies)
        
        smiles_alphabet=list(OrderedSet(''.join(smiles_list)))
        largest_smiles_len=len(max(smiles_list, key=len))
        selfies_len=[]
        
        print(f'--> Building alphabets for {smiles_list.shape[0]} smiles and selfies in dataset...')
        for individual_selfie in selfies_list:
            selfies_len.append(len(individual_selfie)-len(individual_selfie.replace('[',''))) # len of SELFIES
        selfies_alphabet_pre=list(OrderedSet(''.join(selfies_list)[1:-1].split('][')))
        selfies_alphabet=[]
        for selfies_element in selfies_alphabet_pre:
            selfies_alphabet.append('['+selfies_element+']')        
        largest_selfies_len=max(selfies_len)
        
        print('Finished parsing smiles and selfies alphabet. Saving to pickle file custom_alphabets.pickle')
        print('Longest selfies : ',  largest_selfies_len)
        print('Longest smiles : ',  largest_smiles_len)
        
        d= {'selfies_alphabet':selfies_alphabet,
            'largest_selfies_len':largest_selfies_len,
            'smiles_alphabet': smiles_alphabet,
            'largest_smiles_len': largest_smiles_len}
        
        with open('custom_alphabets.pickle', 'wb') as f :
            pickle.dump(d,f)
        
        return (selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len)
    
    def selfies_to_hot(self, molecule):
        """
        Go from a single selfies string to a list of integers
        """
        # integer encode input smile
        len_of_molecule=len(molecule)-len(molecule.replace('[',''))
        for _ in range(self.max_len - len_of_molecule): # selfies padding 
            molecule+='[epsilon]'
    
        selfies_char_list_pre=molecule[1:-1].split('][')
        selfies_char_list=[]
        for selfies_element in selfies_char_list_pre:
            selfies_char_list.append('['+selfies_element+']')   
    
        try:
            integer_encoded = [self.char_to_index[char] for char in selfies_char_list]
            a = np.array(integer_encoded)
            valid_flag = 1 
        except:
            a = 0
            valid_flag = 0  # no one hot encoding possible : ignoring molecule 
                
        return a, valid_flag

    def __getitem__(self, idx):
        # Returns tuple 
        # Smiles has to be in first column of the csv !!

        row = self.df.iloc[idx,:]
        
        if self.input_type =='smiles':
            smiles = row.smiles # needed anyway to build graph 
        elif self.input_type =='selfies':
            selfies = row.selfies # needed anyway to build graph 
            smiles = decoder(selfies)
            
        w = row.weights
        
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
            print('!!!! Atom type to one-hot error for input ', smiles, ' ignored')
            return None, 0,0,0

        at_charge = {a: oh_tensor(self.charges_map[label], self.num_charges) for a, label in
                     (nx.get_node_attributes(graph, 'formal_charge')).items()}
        nx.set_node_attributes(graph, name='formal_charge', values=at_charge)
        
        hydrogens = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                   (nx.get_node_attributes(graph, 'num_explicit_hs')).items()}
        nx.set_node_attributes(graph, name='num_explicit_hs', values=hydrogens)
        
        aromatic = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                   (nx.get_node_attributes(graph, 'is_aromatic')).items()}
        nx.set_node_attributes(graph, name='is_aromatic', values=aromatic)

        at_chir = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                   (nx.get_node_attributes(graph, 'chiral_tag')).items()}
        nx.set_node_attributes(graph, name='chiral_tag', values=at_chir)

        # to dgl 
        g_dgl = dgl.DGLGraph()
        node_features = ['atomic_num', 'formal_charge', 'num_explicit_hs', 'is_aromatic', 'chiral_tag']
        g_dgl.from_networkx(nx_graph=graph,
                            node_attrs=node_features,
                            edge_attrs=['one_hot']) 
        
        N=g_dgl.number_of_nodes()

        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata[f].view(N,-1) for f in node_features], dim=1)

        # 2 - Smiles / selfies to integer indices array
        if self.language == 'selfies': # model works with selfies
            
            if self.input_type == 'smiles': # input to dataloader is smiles// get kekulesmiles selfie
                m=Chem.MolFromSmiles(smiles)
                Chem.Kekulize(m)
                string_representation = encoder(Chem.MolToSmiles(m, kekuleSmiles=True))
                
            elif self.input_type =='selfies':
                string_representation = selfies
            
            a, valid_flag = self.selfies_to_hot(string_representation)
            
            if valid_flag ==0 and self.debug : # no one hot encoding for this selfie, ignore 
                print('!!! Selfie to one-hot failed with current alphabet:')
                print(smiles)
                return None, 0,0
            
        else: # model decodes to smiles 
            string_representation = smiles
            a = np.zeros(self.max_len)
            idces = [self.char_to_index[c] for c in string_representation]
            a[:len(idces)] = idces

        return g_dgl, a, w 

if __name__ == '__main__':
     pass
