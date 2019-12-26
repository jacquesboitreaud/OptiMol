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

def pca_plot(z):
    """ Fits PCA on latent batch z (N*latent_dim) """
    pca = PCA(n_components=2)
    z2 = pca.fit_transform(z)
    
    sns.scatterplot(x=z2[:,0], y=z2[:,1], s=15)
    plt.title("Latent embeddings visualized in 2D PCA space")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show