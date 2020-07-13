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

rc = {'figure.figsize': (10, 5),
      'axes.facecolor': 'white',
      'axes.grid': True,
      'lines.linewidth': 4,
      'grid.color': '.8',
      'font.size': 12}
plt.rcParams.update(rc)


def plot_csvs(dir_paths, ylim=(-12, -6), plot_best=False, return_best=False, use_norm_score=False, obj='logp',
              successive=True):
    """
    get scores in successive dfs.
    Expect the names to be in format *_iteration.csv
    Expect to contain a 'score' column to get mean and std over
    :param ylim : scores range for plt plot
    :param plot_best: show best score for each iter with a red dot on the errorbars plot
    :param return_best: returns a list of best smiles with their scores 
    :param path:
    :return:
    """

    def plot_one(dir_path, use_norm_score=False, obj='logp', successive=successive):
        # 2,3,23 not 2,23,3
        names = os.listdir(dir_path)
        numbers = [int(name.split('_')[-1].split('.')[0]) for name in names]
        asort = np.argsort(np.array(numbers))
        iterations = np.array(numbers)[asort][:21]
        sorted_names = np.array(names)[asort][:21]

        batch_size = None  # default
        mus, stds, best, best_smiles = list(), list(), list(), list()
        olds = set()
        newslist = list()
        for name in sorted_names:
            # Check scores
            news = 0
            df = pd.read_csv(os.path.join(dir_path, name))
            df = df[df['score'] != 0]
            if use_norm_score:
                values = df['norm_score']
            else:
                values = df['score']
            mus.append(np.mean(values))
            stds.append(np.std(values))
            if obj == 'docking':  # lowest docking score is better
                i_best = np.argmin(values)
                best.append(np.min(values))
            else:
                i_best = np.argmax(values)
                best.append(np.max(values))

            # Check novelty
            smiles = df['smile']
            # print(values[3])
            # print(smiles)
            best_smiles.append((smiles[i_best], values[i_best]))
            for smile in smiles:
                if smile not in olds:
                    olds.add(smile)
                    news += 1
            newslist.append(news)
            if successive:
                olds = set(smiles)
        if batch_size is None:
            batch_size = 1000  # default
        newslist = [min(batch_size, new_ones) for new_ones in newslist]
        title = dir_path.split("/")[-1]

        return iterations, mus, stds, batch_size, newslist, title, best, best_smiles

    if not isinstance(dir_paths, list):  # plot only one cbas
        print(dir_paths)
        fig, ax = plt.subplots(1, 2)
        iterations, mus, stds, batch_size, newslist, title, best_scores, best_smiles = plot_one(dir_paths,
                                                                                                use_norm_score,
                                                                                                obj,
                                                                                                successive=successive)
        print(newslist)
        mus = np.array(mus)
        stds = np.array(stds)
        ax[0].fill_between(iterations, mus + stds, mus - stds, alpha=.25)
        sns.lineplot(iterations, mus, ax=ax[0])

        if plot_best:
            ax[0].plot(iterations, best_scores, 'r.')
        ax[0].set_ylim(ylim[0], ylim[1])
        ax[0].set_xlim(1, iterations[-1] + 0.2)

        ax[1].set_ylim(0, batch_size + 100)
        ax[1].plot(iterations, newslist)
        sns.despine()
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('Docking Score (kcal/mol)')
        ax[1].set_xlabel('Iterations')
        ax[1].set_ylabel('Novel samples')
        fig.tight_layout(pad=2.0)
        fig.align_labels()
    else:  # plot multiple
        fig, ax = plt.subplots(2, len(dir_paths))
        for i, dir_path in enumerate(dir_paths):
            iterations, mus, stds, batch_size, newslist, title, best_scores, best_smiles = plot_one(dir_path,
                                                                                                    use_norm_score,
                                                                                                    obj,
                                                                                                    successive=successive)
            mus = np.array(mus)
            stds = np.array(stds)
            ax[0, i].fill_between(iterations, mus - stds, mus + stds, alpha=.25)
            sns.lineplot(iterations, mus, ax=ax[0, i])

            if plot_best:
                ax[0, i].plot(iterations, best_scores, 'r*')
            ax[0, i].set_ylim(*ylim)
            ax[0, i].set_xlim(1, iterations[-1] + 0.5)
            # ax[0, i].set_title(title)

            ax[1, i].set_ylim(0, batch_size + 100)
            ax[1, i].plot(iterations, newslist)

    plt.savefig("cbas_fig.pdf", format="pdf")
    plt.show()

    if return_best:
        return best_smiles


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
    pass
    # plot_csvs('plot/long_gaussian')
    # plot_csvs('plot/gpu_test')
    # plot_csvs('plot/sgd')
    # plot_csvs('plot/sgd_large')
    # plot_csvs('plot/more_epochs')
    # plot_csvs(['plot/more_lr', 'plot/gamma'])
    # plot_csvs('plot/more_both')
    # plot_csvs('plot/clogp_adam_small')
    # plot_csvs('plot/qed_ln_nosched_big_lesslr', ylim=(0.5, 1))
    # plot_csvs(['plot/robust_run2','plot/qed_ln_nosched_big'], ylim=(0.5, 1))
    # plot_csvs(['plot/big_newlr2','plot/big_lnnosched'])
    # plot_csvs('plot/big_lnnosched')
    # plot_csvs('plot/big_newlr2', successive=False)
    # plot_csvs('plot/clogp_adam_clamp_avged', plot_best=True)
    # plot_csvs(['plot/zinc1', 'plot/zinc2'])
    # plot_csvs(['plot/large_samples', 'plot/gpu_test'])
    # plot_csvs('plot/gamma_lr')
    # plot_csvs('plot/adam_nosched')
    # plot_csvs('plot/multi')
    # plot_csvs('plot/big_newlr2')
    plot_csvs('plot/multi',successive=False)
