# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 11:25:49 2019

@author: jacqu
"""

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import numpy as np 

from sklearn.decomposition import PCA 

from rdkit import Chem
from rdkit.Chem import Draw


# ==================== PCA plots ====================================
def pca_plot(z):
    """ Fits PCA on latent batch z (N*latent_dim) """
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)
    
    sns.scatterplot(x=z2[:,0], y=z2[:,1], s=15)
    plt.title("Latent embeddings visualized in 2D PCA space")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show
    
def pca_plot_true_affs(z, affs):
    """ Fits PCA on latent batch z (N * ldim) and plots colored according to target preference of molecules """
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)
    
    
    sns.scatterplot(x=z2[:,0], y=z2[:,1], s=15, hue = affs)
    plt.title("Latent embeddings visualized in 2D PCA space")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show
    
    
# ================== Molecules drawing  ===============================
    
def draw_mols(df, col='output smiles', cutoff = 40):
    # RDKIT drawing of smiles in a dataframe
    smiles = list(df[col])
    
    mols=[Chem.MolFromSmiles(s.rstrip('\n')) for s in smiles]
    
    mols= [m for m in mols if m!=None]
    print(len(mols))
    img = Draw.MolsToGridImage(mols, legends=[str(i) for i in range(len(mols))])
    
    return img, mols

    
    