# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:35:56 2020

@author: jacqu

Oracle functions for molecules (given as SMILES strings)
"""

from rdkit import Chem
from rdkit.Chem import QED

import torch
import numpy as np


def qed(smiles):
    # takes a list of smiles and returns a list of corresponding QEDs
    t = torch.zeros(len(smiles))
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if (m != None):
            t[i] = QED.qed(m)
    return t


def isValid(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m == None:
        return 0
    return 1


def normal_cdf_oracle(observed_x, var, gamma):
    """
    Assuming x ~ N(observed_x, var), returns P(x<=gamma)
    """
    raise NotImplementedError


def deterministic_cdf_oracle(observed_x, gamma):
    """
    Returns P(x<= gamma) assuming x is equal to observed_x with proba 1 
    """
    w = torch.zeros(observed_x.shape[0], dtype=torch.float)
    for i in range(observed_x.shape[0]):
        w[i] = deterministic_one(observed_x[i], gamma)
    return w


def deterministic_one(value, gamma):
    return float(value <= gamma)
