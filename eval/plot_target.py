# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:19:15 2020

@author: jacqu

Plotting actives of different targets in latent space of preloaded model.
"""
import numpy as np
import pandas as pd

import os 

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    return


if(__name__=='__main__'): 
    
    # Random ZINC // no need to re-embed if already done 
    #pca_plot_color(z= z_r, pca = fitted_pca, color = 'lightblue', label='random ZINC')


    p = sns.color_palette()

    tar = 'vgfr2'
    df_a = pd.read_csv(f'../data/targets/{tar}+.csv')
    print(df_a.shape)
    
    z_a = model.embed(loaders, df_a)
    
    fitted_pca = load('fitted_pca.joblib') 
    #fitted_pca = fit_pca(z_all)
    #plt.xlim(-4.1,4.1)
    #plt.ylim(-2.8,2.4)
        
    pca_plot_color(z= z_all, pca = fitted_pca, color='lightblue', label = 'random ZINC')
    pca_plot_color(z= z_a, pca = fitted_pca, color = 'green', label = f'{tar} actives')
    
    
    # Pairwise distances 
    #for i in range(len(Z)):
        #D_a = pairwise_distances(z_all, metric='cosine')
        #print(np.mean(D_a))