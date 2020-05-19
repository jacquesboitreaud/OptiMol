# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:19:42 2020

@author: jacqu

Cycle scorer for composite QED and composite logP
"""


import networkx as nx

from rdkit import Chem
from rdkit.Chem import rdmolops

def cycle_score(smi):
    """
    Input : a smiles string
    Output : cycle score penalty (scalar)
    """
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi))))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
        
    return -1*cycle_length

