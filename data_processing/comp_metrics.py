# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:19:42 2020

@author: jacqu

Functions to compute composite QED and composite logP
"""


import networkx as nx
import os, sys


from rdkit import Chem
from rdkit.Chem import rdmolops, QED, Crippen
script_dir_metrics = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir_metrics)
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
        
    return float(cycle_length)

def logP(m):
    return Crippen.MolLogP(m)

def qed(m):
    return Chem.QED.qed(m)


def cLogP(smi, errorVal = -20):
    """
    Input : a smiles string
    error val : value to return for invalid smiles 
    Output : composite logP (scalar)
    """
    
    m=Chem.MolFromSmiles(smi)
    
    if m is None:
        return errorVal
    else:
        logP = Chem.Crippen.MolLogP(m) # logp high => good. what we look for 
        c = cycle_score(m) # c big => not good // big cycles 
        s = calculateScore(m) # s big => not good // hard to make 
        
    return logP - s - c

def cQED(smi, errorVal = -20):
    """
    Input : a smiles string
    Output : composite logP (scalar)
    """
    
    m=Chem.MolFromSmiles(smi)
    
    if m is None:
        return errorVal
    else:
        q = Chem.QED.qed(m)
        c = cycle_score(m)
        s = calculateScore(m)
        
    return q - s - c
        
    
if __name__=='__main__':
    
    s='CC1=C(Br)C=CC=C1NC(=O)CN1CC=C2C3CCOC(=O)C(C)C2CCCC3NC1=O'
    
    q = cQED(s)
    p =cLogP(s)
    
    print('Penalized logP: ', p)
    
    
