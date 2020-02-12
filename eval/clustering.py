# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:19:15 2020

@author: jacqu

PCA plot of embeddings and clustering  in latent space (TODO)
"""
import numpy as np
import pandas as pd

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

    # Extend for either 1 or 2 targets 
    
    target = 'herg+'
    target2 = 'herg-'
    
    fold = 1
    # Actives 
    df_a = pd.read_csv(f'data/targets/herg+.csv')
    df_a = df_a[df_a['fold']==fold]
    
    if(target2!=None):
        df_a2 = pd.read_csv(f'data/targets/drd3+.csv')
        df_a2 = df_a2[df_a2['fold']==fold]
    
    
    # Random ZINC 
    df_r = pd.read_csv(f'../data/moses_test.csv')
    df_r = df_r.sample(10000)
    
    
    z_a = embed(model,device, loaders, df_a)
    if(target2 != None):
        z_a2 = embed(model,device, loaders, df_a2)
    z_r = embed(model,device, loaders, df_r)
    
    
    # Plot in PCA space 
    fitted_pca = load('fitted_pca.joblib') 
    
    # Refit pca ? 
    z_prime = np.concatenate((z_a,z_a2))
    fitted_pca=fit_pca(z_prime)
    
    plt.figure()

    pca_plot_color(z= z_r, pca = fitted_pca, color = 'lightblue', label='random ZINC') # random moses
    pca_plot_color(z= z_a, pca = fitted_pca, color = 'green', label = 'cluster 1') #actives 1 
    pca_plot_color(z= z_a2, pca = fitted_pca, color = 'red', label = 'cluster 2') # actives 2

    # Pairwise distances 
    D_a = pairwise_distances(z_a, metric='l2')
    D_d = pairwise_distances(z_a2, metric='l2')
    
    """
    #avg_a, avg_d = np.mean(D_a), np.mean(D_d)
    hclust = AgglomerativeClustering(distance_threshold=0.11,n_clusters = None,
                                        linkage="average", affinity='precomputed')
    hclust=hclust.fit(D_a)
    plt.figure()
    plot_dendrogram(hclust,truncate_mode='level', p=1)
    plt.show()
    """
