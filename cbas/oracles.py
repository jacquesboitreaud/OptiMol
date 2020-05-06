# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:35:56 2020

@author: jacqu

Oracle functions for molecules (given as SMILES strings)
"""

from rdkit import Chem
from rdkit.Chem import QED

import torch

def qed(smiles):
    # takes a list of smiles and returns a list of corresponding QEDs
    t = torch.zeros(len(smiles))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if (m != None):
            t[i] = QED.qed(m)
    return t

def isValid(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m==None:
        return 0
    return 1

def normal_cdf(observed_x, var, gamma):
    """
    Assuming x ~ N(observed_x, var), returns P(x<=gamma)
    """
    raise NotImplementedError
    
def certain_cdf(observed_x, gamma):
    """
    Returns P(x<= gamma) assuming x is equal to observed_x with proba 1 
    """
    if observed_x <= gamma:
        return 1.
    else:
        return 0 
