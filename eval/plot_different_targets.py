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
    
    tars = os.listdir('../data/targets/gpcr')
    tars = [t for t in tars if '+' in t ]
    p = sns.color_palette()
    
    Z=[]
    labels = []
    
    for i,target_f in enumerate(tars) :
        target= target_f[:-5]
        print(target)
    
        df_a = pd.read_csv(f'../data/targets/gpcr/{target_f}', index_col = 0)
        print(df_a.shape)
        N = df_a.shape[0]
        k = int(N/2)
        df_a=df_a.iloc[:k,:]
    
        # Select a split fold of the data or a subsample 
        # df_a = df_a[df_a['fold']==1]
        
        fitted_pca = load('fitted_pca.joblib') 
    
        z_a = embed(model,device, loaders, df_a)
        labels.append(np.ones(z_a.shape[0])*i)
        Z.append(z_a)
        
    z_all = np.concatenate(Z)
    lab_all = np.concatenate(labels)
    
    #fitted_pca = fit_pca(z_all)
        
    for i,z in enumerate(Z) :
        target = tars[i][:-5]
        pca_plot_color(z= z, pca = fitted_pca, color = p[i], label = f'{target}') 
    
    
    # Pairwise distances 
    for i in range(len(Z)):
        D_a = pairwise_distances(Z[i], metric='cosine')
        print(np.mean(D_a))
    #D_d = pairwise_distances(z_d, metric='l2')
    
    """
    #avg_a, avg_d = np.mean(D_a), np.mean(D_d)
    hclust = AgglomerativeClustering(distance_threshold=1,n_clusters = None,
                                        linkage="average", affinity='precomputed')
    hclust=hclust.fit(D_a)
    plt.figure()
    plot_dendrogram(hclust,truncate_mode='level', p=10)
    plt.show()
    """

s= silhouette_score(z_all, lab_all, metric='cosine')
print('silhouette score ', s)