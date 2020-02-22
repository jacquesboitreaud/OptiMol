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
    
    profiles = ['aa2ar', 'adrb1/2','drd3']
    p = sns.color_palette()
    
    Z=[]
    labels = []
    fold = 2
    
    for i,prof in enumerate(profiles) :
    
        df_a = pd.read_csv(f'../data/exp/gpcr/gpcr_3cl.csv')
        print('selecting fold ', fold)
        df_a=df_a[df_a['fold']==fold]
        
        df_a =df_a[df_a['profile']==prof]
        print(prof, df_a.shape[0])
        
        z_a = model.embed( loaders, df_a)
        labels.append(np.ones(z_a.shape[0])*i)
        Z.append(z_a)
        
    z_all = np.concatenate(Z)
    lab_all = np.concatenate(labels)
    
    fitted_pca = load('fitted_pca.joblib') 
    
    fitted_pca = fit_pca(z_all)
    #plt.xlim(-4.1,4.1)
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
    # Inverse transform centroids for sampling 
    c_drd3 = [2,-1]
    c_adrb = [-1.8,-1.5]
    c_aa2ar = [3,0]
    z_cdrd3 = torch.FloatTensor(fitted_pca.inverse_transform(c_drd3)).view(1,-1)
    z_adrb = torch.FloatTensor(fitted_pca.inverse_transform(c_adrb)).view(1,-1)
    z_aa2ar = torch.FloatTensor(fitted_pca.inverse_transform(c_aa2ar)).view(1,-1)
    
    with open('../generate/centroids_ft.pickle', 'wb') as f:
        pickle.dump(z_cdrd3,f)
        pickle.dump(z_adrb,f)
        pickle.dump(z_aa2ar,f)
    # Sampling : 
    drd3_sp = model.sample_around_z(z_cdrd3, dist=3, beam_search=False, attempts=1000)
    """

s= silhouette_score(z_all, lab_all, metric='cosine')
print('silhouette score ', s)