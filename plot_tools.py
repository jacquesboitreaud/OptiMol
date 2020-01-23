# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:25:49 2019

@author: jacqu
"""

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import numpy as np 



from rdkit import Chem
from rdkit.Chem import Draw



    
# ================== Molecules drawing  ===============================
    
def draw_mols(df, col='output smiles', cutoff = 40):
    # RDKIT drawing of smiles in a dataframe
    smiles = list(df[col])
    
    mols=[Chem.MolFromSmiles(s.rstrip('\n')) for s in smiles]
    
    mols= [m for m in mols if m!=None]
    print(len(mols))
    img = Draw.MolsToGridImage(mols, legends=[str(i) for i in range(len(mols))])
    
    return img, mols

    
    