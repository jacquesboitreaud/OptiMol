# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:14:21 2020

@author: jacqu

Parsing BO results in ./results 

Code from Kusner et al. in Grammar VAE 

@ https://github.com/mkusner/grammarVAE 
"""

import pickle
import gzip
import numpy as np

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

import matplotlib.pyplot as plt
import seaborn as sns

# Params 

name = 'big_run'
n_simulations = 1
iteration = 20
# We define the functions used to load and save objects

def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret

# We compute the average statistics for the grammar autoencoder
    
results_grammar = np.zeros((n_simulations, 3))
for j in range(1, n_simulations + 1):
    best_value = 1e10
    n_valid = 0
    max_value = 0
    for i in range(iteration):
        smiles = load_object(f'results/{name}/simulation_{j}/valid_smiles_{i}.dat')
        scores = load_object(f'results/{name}/simulation_{j}/scores_{i}.dat')
        n_valid += len([ x for x in smiles if x is not None ])

        if min(scores) < best_value:
            best_value = min(scores)
        if max(scores) > max_value:
            max_value = max(scores)

    

    sum_values = 0
    count_values = 0
    for i in range(iteration):
        scores = np.array(load_object(f'results/{name}/simulation_{j}/scores_{i}.dat'))
        sum_values += np.sum(scores[  scores < max_value ])
        count_values += len(scores[  scores < max_value ])
        
    # fraction of valid smiles

    results_grammar[ j - 1, 0 ] = 1.0 * n_valid / (iteration * 50)

    # Best value

    results_grammar[ j - 1, 1 ] = best_value

    # Average value = 

    results_grammar[ j - 1, 2 ] = 1.0 * sum_values / count_values

print("Results (fraction valid, best, average)):")

print("Mean:", np.mean(results_grammar, 0)[ 0 ], -np.mean(results_grammar, 0)[ 1 ], -np.mean(results_grammar, 0)[ 2 ])
print("Std:", np.std(results_grammar, 0) / np.sqrt(iteration))


print("First:", -np.min(results_grammar[ : , 1 ]))
best_score = np.min(results_grammar[ : , 1 ])
results_grammar[ results_grammar[ : , 1 ] == best_score , 1 ] = 1e10
print("Second:", -np.min(results_grammar[ : , 1 ]))
second_best_score = np.min(results_grammar[ : , 1 ])
results_grammar[ results_grammar[ : , 1 ] == second_best_score, 1 ] = 1e10
print("Third:", -np.min(results_grammar[ : , 1 ]))
third_best_score = np.min(results_grammar[ : , 1 ])

mols = []
for j in range(1, n_simulations + 1):
    for i in range(iteration):
        smiles = np.array(load_object(f'results/{name}/simulation_{j}/valid_smiles_{i}.dat'))
        scores = np.array(load_object(f'results/{name}/simulation_{j}/scores_{i}.dat'))
        
        if np.any(scores == best_score):
            smile = smiles[ scores == best_score ]
            smile = np.array(smile).astype('str')[ 0 ]
            print("First:", smile)
            mol = MolFromSmiles(smile)
            mols.append(mol)
            best_score = 1e10

        if np.any(scores == second_best_score):
            smile = smiles[ scores == second_best_score ]
            smile = np.array(smile).astype('str')[ 0 ]
            print("Second:", smile)
            mol = MolFromSmiles(smile)
            mols.append(mol)
            second_best_score = 1e10

        if np.any(scores == third_best_score):
            smile = smiles[ scores == third_best_score ]
            smile = np.array(smile).astype('str')[ 0 ]
            print("Third:", smile)
            mol = MolFromSmiles(smile)
            mols.append(mol)
            third_best_score = 1e10

img = Draw.MolsToGridImage(mols, molsPerRow = len(mols), subImgSize=(300, 300), useSVG=True)
with open(f"results/{name}/best_molecule.svg", "w") as text_file:
    text_file.write(img)
    
# Distribution plots 


for j in range(1, n_simulations + 1):
    means, stds , best = [], [], []
    
    for i in range(iteration):
        smiles = np.array(load_object(f'results/{name}/simulation_{j}/valid_smiles_{i}.dat'))
        scores = -np.array(load_object(f'results/{name}/simulation_{j}/scores_{i}.dat'))
        
        means.append(np.mean(scores))
        stds.append(np.std(scores))
        
        best.append(max(scores))
        
    fig, ax = plt.subplots(1,2)
    
    it = np.arange(iteration)+1
    mus = np.array(means)
    stds = np.array(stds)
    ax[0].fill_between(it, mus+stds, mus-stds, alpha=.25)
    sns.lineplot(it, mus, ax=ax[0])
    ax[0].plot(it, best, 'r.')
    ax[0].set_xlim(1,it[-1]+0.2)
    
    ax[0].set_ylim(-6.8,11.5)
    plt.xlabel('Step')
    #plt.ylabel('Score')
    sns.despine()
    plt.show()
    
    