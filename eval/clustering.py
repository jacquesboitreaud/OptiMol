# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:19:15 2020

@author: jacqu

Clustering molecules given their 
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
    
    target = 'herg'
    target2 = 'drd3'
    
    # Actives 
    df_a = pd.read_csv(f'../data/triplets/{target}_actives.csv')
    df_a = df_a[df_a['fold']==2]
    
    if(target2!=None):
        df_a2 = pd.read_csv(f'../data/triplets/{target2}_actives.csv')
        df_a2 = df_a2[df_a2['fold']==2]
    
    # Decoys 
    df_d = pd.read_csv(f'../data/triplets/{target}_decoys.csv')
    df_d = df_d.sample(5000)
    
    # Random ZINC 
    df_r = pd.read_csv(f'../data/moses_test.csv')
    df_r = df_r.sample(5000)
    
    
    z_a = embed(model,device, loaders, df_a)
    if(target2 != None):
        z_a2 = embed(model,device, loaders, df_a2)
    z_d = embed(model,device,loaders, df_d)
    z_r = embed(model,device, loaders, df_r)
    
    
    # Plot in PCA space 
    fitted_pca = load('fitted_pca.joblib') 
    plt.figure()

    pca_plot_color(z= z_r, pca = fitted_pca, color = 'red') # random moses
    pca_plot_color(z= z_d, pca = fitted_pca, color = 'lightgreen') # decoys
    pca_plot_color(z= z_a, pca = fitted_pca, color = 'purple') #actives 1 
    if(target2 != None):
        pca_plot_color(z= z_a2, pca = fitted_pca, color = 'blue') # actives 2

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
