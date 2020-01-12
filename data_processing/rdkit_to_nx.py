# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:05:02 2019

@author: jacqu

Another solution for smiles -> nx graph, but taking into account the stereochemistry.
Maybe slower 

"""

import networkx as nx
import argparse
import multiprocessing
from rdkit import Chem
import torch

import dgl

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
                   
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G

def smiles_to_nx(smiles, validate=False):
    mol = Chem.MolFromSmiles(smiles)
    G = mol_to_nx(mol)
    return G

def nx_to_mol(G, edge_map, at_map, chi_map, charge_map):
    if(type(G)!=nx.classes.graph.Graph):
        G=G.to_networkx(node_attrs=['atomic_num','chiral_tag','formal_charge','num_explicit_hs','is_aromatic'], 
                    edge_attrs=['one_hot'])
        G=G.to_undirected()
    
    # Map back one-hot encoded node features to actual values ! 
    atomic_nums={i: at_map[torch.argmax(v).item()] for (i,v) in nx.get_node_attributes(G, 'atomic_num').items()}
    chiral_tags={i: chi_map[torch.argmax(v).item()] for (i,v) in nx.get_node_attributes(G, 'chiral_tag').items()}
    formal_charges={i: charge_map[torch.argmax(v).item()] for (i,v) in nx.get_node_attributes(G, 'formal_charge').items()}
    
    bond_types={i: edge_map[v.item()] for (i,v) in nx.get_edge_attributes(G, 'one_hot').items()}
    
    mol = Chem.RWMol()

    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])

        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        try:
            mol.AddBond(ifirst, isecond, bond_type)
        except(RuntimeError):
            continue
    
    #Uncomment to sanitize molecule
    Chem.SanitizeMol(mol)
    return mol

if __name__ == '__main__':
    pass