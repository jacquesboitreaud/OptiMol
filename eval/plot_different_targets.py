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
from sklearn.metrics import pairwise_distances

from model_utils import load_trained_model, embed


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
    
    tars = os.listdir('../data/targets')
    tars = [t for t in tars if '+' in t ]
    p = sns.color_palette()
    
    for i,target_f in enumerate(tars) :
        target= target_f[:-5]
        print(target)
    
        df_a = pd.read_csv(f'../data/targets/{target_f}', index_col = 0)
    
        # Select a split fold of the data or a subsample 
        # df_a = df_a[df_a['fold']==1]
        
        fitted_pca = load('fitted_pca.joblib') 
    
        z_a = embed(model,device, loaders, df_a)
        
        pca_plot_color(z= z_a, pca = fitted_pca, color = p[i], label = f'{target} actives') 
    
    """
    # Pairwise distances 
    #D_a = pairwise_distances(z_a, metric='l2')
    #D_d = pairwise_distances(z_d, metric='l2')
    
    #avg_a, avg_d = np.mean(D_a), np.mean(D_d)
    hclust = AgglomerativeClustering(distance_threshold=1,n_clusters = None,
                                        linkage="average", affinity='precomputed')
    hclust=hclust.fit(D_a)
    plt.figure()
    plot_dendrogram(hclust,truncate_mode='level', p=10)
    plt.show()
    """
