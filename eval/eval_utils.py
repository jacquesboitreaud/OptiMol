# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:46:47 2020

@author: jacqu
"""
import os
import numpy as np
from numpy.linalg import norm
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols

from sklearn.decomposition import PCA


def plot_csvs(dir_path):
    """
    get scores in successive dfs.
    Expect the names to be in format *_iteration.csv
    Expect to contain a 'score' column to get mean and std over
    :param path:
    :return:
    """
    # 2,3,23 not 2,23,3
    names = os.listdir(dir_path)
    numbers = [int(name.split('_')[-1].split('.')[0]) for name in names]
    asort = np.argsort(np.array(numbers))
    iterations = np.array(numbers)[asort]
    sorted_names = np.array(names)[asort]

    batch_size = None  # default
    mus, stds = list(), list()
    olds = set()
    newslist = list()
    for name in sorted_names:
        # Check scores
        news = 0
        df = pd.read_csv(os.path.join(dir_path, name))
        values = df['score']
        mus.append(np.mean(values))
        stds.append(np.std(values))

        # Check novelty
        smiles = df['smile']
        for smile in smiles:
            if smile not in olds:
                olds.add(smile)
                news += 1
        newslist.append(news)

        if batch_size is None:
            batch_size = len(smiles)
    if batch_size is None:
        batch_size = 1000  # default

    fig, ax = plt.subplots(1, 2, num=1)
    ax[0].errorbar(iterations, mus, stds, linestyle='None', marker='^')
    ax[1].set_ylim(0, batch_size + 100)
    ax[1].plot(iterations, newslist)
    fig.suptitle(dir_path.split("/")[-1])

    plt.show()


def plot_kde(z):
    """
    Input: 
        numpy array of shape N * latent shape
    Output: 
        Plots KDE of each latent dimension, estimated with the N provided samples
    """
    plt.figure()
    # Iterate through all latent dimensions
    for d in range(z.shape[1]):
        # select this latent dim 
        subset = z[:, d]
        # Draw the density plot
        sns.distplot(subset, hist=False, kde=True,
                     kde_kws={'linewidth': 1})

    # Plot formatting
    plt.title(f'KDE estimate for each latent dimension (n={z.shape[1]})')
    plt.xlabel('Z (unstandardized)')
    # plt.xlim((-0.3,0.3))
    plt.ylabel('Density')


def similarity(smi1, smi2):
    """ Computes tanimoto similarity between two smiles with RDKIT """
    m1, m2 = Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)
    # Check at least one molecule is valid
    if m1 == None or m2 == None:
        return 0
    # here choose the type of fingerprint to usee
    f1, f2 = AllChem.GetMorganFingerprint(m1, 2), AllChem.GetMorganFingerprint(m2, 2)
    # f1, f2 = FingerprintMols.FingerprintMol(m1), FingerprintMols.FingerprintMol(m2)


# ==================== PCA ====================================

def fit_pca(z):
    # Fits pca and returns 
    pca = PCA(n_components=2)
    pca = pca.fit(z)

    return pca


def pca_plot_color(z, pca, color, label):
    """ Takes fitted PCA and applies on latent batch z (N*latent_dim) """
    z2 = pca.transform(z)
    sns.scatterplot(x=z2[:, 0], y=z2[:, 1], s=10, color=color, label=label)
    # plt.title("Latent embeddings visualized in 2D PCA space")
    plt.legend()
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')


def pca_plot_hue(z, pca, variable, label):
    """ Applies fitted PCA on latent batch z (N * ldim) and plots colored according to variable """
    z2 = pca.transform(z)

    palettes = ['YlGnBu', 'GnBu', 'GnBu_r', 'BuGn', 'Blues', 'YlOrRd_r']

    chosen = 'YlGnBu_r'

    ax = sns.scatterplot(x=z2[:, 0], y=z2[:, 1], s=15, hue=variable, palette=chosen, label=label)
    # plt.title("Latent embeddings visualized in 2D PCA space")
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    # Fixing the issue of too many decimals in legend
    leg = ax.legend_
    for i, t in enumerate(leg.texts):
        # truncate label text to 4 characters
        if i > 0:
            t.set_text(t.get_text()[:4])
    return ax


if __name__ == '__main__':
    # plot_csvs('plot/long_gaussian')
    # plot_csvs('plot/gpu_test')
    # plot_csvs('plot/sgd')
    plot_csvs('plot/sgd_large')
    plot_csvs('plot/more_epochs')
    plot_csvs('plot/more_lr')
