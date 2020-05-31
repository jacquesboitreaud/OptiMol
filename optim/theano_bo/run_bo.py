# -*- coding: utf-8 -*-
"""
Created on Sun May 31 19:33:53 2020

@author: jacqu

Bayesian optimization following implementation by Kusner et al, in Grammar VAE 

@ https://github.com/mkusner/grammarVAE 

"""

import os, sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '../..'))

import pickle
import gzip
import numpy as np

import torch

from rdkit import Chem
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

import networkx as nx
from rdkit.Chem import rdmolops

import data_processing.sascorer as sascorer
import data_processing.comp_metrics

from model import model_from_json
from dataloaders.molDataset import Loader
from selfies import encoder,decoder

from sparse_gp import SparseGP
import scipy.stats as sps

# Params 
random_seed = 1 # random seed 

model_name = '250k'
alphabet = '250k_alphabets.json'


# Helper functions used to load and save objects

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


np.random.seed(random_seed)

# We load the data

X = np.loadtxt('../../data/latent_features_and_targets/latent_features.txt')
y = -np.loadtxt('../../data/latent_features_and_targets/targets.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

# Loading the model : 
        
# Loader for initial sample
loader = Loader(props=[], 
                targets=[], 
                csv_path = None,
                maps_path = '../../map_files',
                alphabet_name = alphabet,
                vocab='selfies', 
                num_workers = 0,
                test_only=True)

# Load model (on gpu if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu' # the model device 
gp_device =  'cpu' #'cuda' if torch.cuda.is_available() else 'cpu' # gaussian process device 
model = model_from_json(model_name)
model.to(device)
model.eval()



iteration = 0

# ============ Iter loop ===============
while iteration < 5:

    # We fit the GP

    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
        y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print('Test RMSE: ', error)
    print('Test ll: ', testll)

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print('Train RMSE: ', error)
    print('Train ll: ', trainll)

    # We load the decoder to obtain the molecules




    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))
    
    # We decode the 50 smiles: 
    # Decode z into smiles
    with torch.no_grad():
        gen_seq = model.decode(torch.from_numpy(next_inputs))
        smiles = model.probas_to_smiles(gen_seq)
        valid_smiles_final = []
        for s in smiles :
            s = decoder(s)
            m = Chem.MolFromSmiles(s)
            if m is None : 
                valid_smiles_final.append(None)
            else:
                Chem.Kekulize(m)
                s= Chem.MolToSmiles(m, kekuleSmiles = True)
                valid_smiles_final.append(s)


    new_features = next_inputs

    save_object(valid_smiles_final, f"results/valid_smiles_{iteration}.dat")

    logP_values = np.loadtxt('../../latent_features_and_targets_grammar/logP_values.txt')
    SA_scores = np.loadtxt('../../latent_features_and_targets_grammar/SA_scores.txt')
    cycle_scores = np.loadtxt('../../latent_features_and_targets_grammar/cycle_scores.txt')
    
    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized


    scores = []
    for i in range(len(valid_smiles_final)):
        if valid_smiles_final[ i ] is not None:
            current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
            current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i ]))
            cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
            if len(cycle_list) == 0:
                cycle_length = 0
            else:
                cycle_length = max([ len(j) for j in cycle_list ])
            if cycle_length <= 6:
                cycle_length = 0
            else:
                cycle_length = cycle_length - 6

            current_cycle_score = -cycle_length
         
            current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
            current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
            current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

            score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
        else:
            score = -max(y)[ 0 ]

        scores.append(-score)
        print(i)

    print(valid_smiles_final)
    print(scores)

    save_object(scores, "results/scores{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
    
    print(iteration)
