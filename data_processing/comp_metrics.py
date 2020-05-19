# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:19:42 2020

@author: jacqu

Functions to compute composite QED and composite logP
"""


import networkx as nx

from rdkit import Chem
from rdkit.Chem import rdmolops, QED, Crippen

from sascorer import *

def cycle_score(m):
    """
    Input : a mol object
    Output : cycle score penalty (scalar)
    """
    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(m)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([ len(j) for j in cycle_list ])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
        
    return -1*cycle_length


def cLogP(smi):
    """
    Input : a smiles string
    Output : composite logP (scalar)
    """
    
    m=Chem.MolFromSmiles(smi)
    
    if m is None:
        return 0.0
    else:
        logP = Chem.Crippen.MolLogP(m)
        c = cycle_score(m)
        s = calculateScore(m)
        
    return logP - s - c

def cQED(smi):
    """
    Input : a smiles string
    Output : composite logP (scalar)
    """
    
    m=Chem.MolFromSmiles(smi)
    
    if m is None:
        return 0.0
    else:
        q = Chem.QED.qed(m)
        c = cycle_score(m)
        s = calculateScore(m)
        
    return q - s - c
        
    
if __name__=='__main__':
    
    s='CC=CC=CC1NCCc2cc(OC)c(OC)c(OC)c21'
    
    q = cQED(s)
    p =cLogP(s)
    
    print('Penalized logP: ', p)
    
    
