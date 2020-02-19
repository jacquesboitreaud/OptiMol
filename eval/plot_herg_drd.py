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
    
    profiles = ['herg blockers','drd3 actives', 'common actives']
    p = sns.color_palette()
    p[2]='red'
    p[1]='lightgreen'
    
    Z=[]
    labels = []
    fold = 2
    
    df = pd.read_csv(f'../data/exp/offtarget/herg_drd.csv')
    
    for i,prof in enumerate(profiles) :
    
        df_a=df[df['profile']==i]
        
        print('selecting fold ', fold)
        df_a=df_a[df_a['fold']==fold]
        print(prof, df_a.shape[0])
        
        z_a = model.embed(loaders, df_a)
        labels.append(np.ones(z_a.shape[0])*i)
        Z.append(z_a)
        
    z_all = np.concatenate(Z)
    lab_all = np.concatenate(labels)
    
    #fitted_pca = load('fitted_pca.joblib') 
    fitted_pca = fit_pca(z_all)
    #plt.xlim(-5.3,6.9)
    #plt.ylim(-2.8,2.4)
        
    for i,z in enumerate(Z) :
        prof = profiles[i]
        try:
            pca_plot_color(z= Z[i], pca = fitted_pca, color = p[i], label = f'{prof}') 
        except:
            pass
    
    
    # Pairwise distances 
    for i in range(len(Z)):
        D_a = pairwise_distances(Z[i], metric='cosine')
        print(np.mean(D_a))
    
    """
    hclust = AgglomerativeClustering(distance_threshold=1,n_clusters = None,
                                        linkage="average", affinity='precomputed')
    hclust=hclust.fit(D_a)
    plt.figure()
    plot_dendrogram(hclust,truncate_mode='level', p=10)
    plt.show()
    """

s= silhouette_score(z_all, lab_all, metric='cosine')
print('silhouette score ', s)