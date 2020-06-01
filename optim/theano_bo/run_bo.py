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
import argparse

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
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore
from selfies import encoder,decoder
from utils import soft_mkdir

from sparse_gp import SparseGP
import scipy.stats as sps

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int ,  default=1) # random seed (simulation id for multiple runs)
parser.add_argument('--model', type = str ,  default='250k') # name of model to use 

parser.add_argument('--obj', type = str ,  default='logp') # objective : logp (composite), qed (composite), qsar or docking

parser.add_argument('--n_iters', type = int ,  default=5) # Number of iterations


args, _ = parser.parse_known_args()

# ===========================================================================


# Params 
random_seed = args.seed # random seed 
soft_mkdir(f'results/simulation_{random_seed}')


model_name = args.model
if '250k' in model_name:
        alphabet = '250k_alphabets.json'
elif 'zinc' in args.name:
    alphabet = 'zinc_alphabets.json'
else:
    alphabet = 'moses_alphabets.json'
print(f'Using alphabet : {alphabet}. Make sure it is coherent with args.model = {args.model}')


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
y = -np.loadtxt('../../data/latent_features_and_targets/targets_{args.obj}.txt')
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
while iteration < args.n_iters:

    # We fit the GP

    np.random.seed(iteration * random_seed)
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
        y_test, minibatch_size = 10 * M, max_iterations = 1, learning_rate = 0.0005)

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

    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))
    
    # We decode the 50 smiles: 
    # Decode z into smiles
    with torch.no_grad():
        gen_seq = model.decode(torch.FloatTensor(next_inputs).to(device))
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
    save_object(valid_smiles_final, f"results/simulation_{random_seed}/valid_smiles_{iteration}.dat")
    
    if args.obj == 'logp':

        logP_values = np.loadtxt('../../data/latent_features_and_targets/logP_values.txt')
        SA_scores = np.loadtxt('../../data/latent_features_and_targets/SA_scores.txt')
        cycle_scores = np.loadtxt('../../data/latent_features_and_targets/cycle_scores.txt')
    
        scores = []
        for i in range(len(valid_smiles_final)):
            if valid_smiles_final[ i ] is not None:
                m= MolFromSmiles(valid_smiles_final[ i ])
                
                current_log_P_value = logP(m)
                current_SA_score = -calculateScore(m)
                current_cycle_score = -cycle_score(m)
                
                # Normalize 
                current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
                current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
                current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
    
                score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
            else:
                score = -max(y)[ 0 ]

            scores.append(-score)
        
    elif args.obj == 'qed':
        
        qed_values = np.loadtxt('../../data/latent_features_and_targets/qed_values.txt')
        SA_scores = np.loadtxt('../../data/latent_features_and_targets/SA_scores.txt')
        cycle_scores = np.loadtxt('../../data/latent_features_and_targets/cycle_scores.txt')
    
        scores = []
        for i in range(len(valid_smiles_final)):
            if valid_smiles_final[ i ] is not None:
                m= MolFromSmiles(valid_smiles_final[ i ])
                
                current_qed_value = qed(m)
                current_SA_score = -calculateScore(m)
                current_cycle_score = -cycle_score(m)
                
                # Normalize 
                current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
                current_qed_value_normalized = (current_qed_value - np.mean(qed_values)) / np.std(qed_values)
                current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)
    
                score = (current_SA_score_normalized + current_qed_value_normalized + current_cycle_score_normalized)
            else:
                score = -max(y)[ 0 ]

            scores.append(-score)
            
    elif args.obj == 'qsar':
        raise NotImplementedError
        
    elif args.obj == 'docking':
        raise NotImplementedError
        
        
    
    # Common to all objectives ; saving scores and smiles for this step 
    print(i)
    print(valid_smiles_final)
    print(scores)

    save_object(scores, f"results/simulation_{random_seed}/scores_{iteration}.dat")

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)

    iteration += 1
    
    print(iteration)
