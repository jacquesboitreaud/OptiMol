# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Fit PCA to moses_test molecules and plots them in latent space . 

"""

import os
import sys
import argparse

script_dir = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import sys
import torch

from rdkit.Chem import Draw

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import torch.utils.data
from torch import nn, optim

from joblib import dump, load
from sklearn.decomposition import PCA

from dataloaders.molDataset import Loader
from data_processing.rdkit_to_nx import smiles_to_nx
from model import Model, model_from_json
from eval.eval_utils import *
from utils import *
from dgl_utils import send_graph_to_device


parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Saved model directory, in /results/saved_models",
                        default='inference_default')
args, _ = parser.parse_known_args()

# ============================

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 100 
# Load model (on gpu if available)
model = model_from_json(args.name)
model.to(device)


# Load eval set
loaders = Loader(csv_path='../data/moses_test.csv',
                 n_mols=-1,
                 num_workers=0,
                 batch_size=batch_size,
                 test_only=True,
                 graph_only=True, 
                 props=[],
                 targets = [])
rem, ram, rchim, rcham = loaders.get_reverse_maps()

_, _, test_loader = loaders.get_data()

# Print model summary
map = ('cpu' if device == 'cpu' else None)
torch.manual_seed(1)

# Validation pass
model.eval()
t_rec, t_mse = 0, 0

# Latent embeddings
z_all = np.zeros((loaders.dataset.n, model.l_size))

with torch.no_grad():
    for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
        if (batch_idx % 100 == 0):
            print(f'{batch_idx*batch_size}/{loaders.dataset.n} molecules passed' )

        graph = send_graph_to_device(graph, device)

        # Latent embeddings
        z = model.encode(graph, mean_only=True)  # z_shape = N * l_size
        z = z.cpu().numpy()
        z_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = z

    # ===================================================================
    # PCA fitting
    # ===================================================================
    fitted_pca = fit_pca(z)
    dump(fitted_pca,  os.path.join(script_dir, f'results/saved_models/{args.name}/fitted_pca.joblib'))
    print('Fitted and saved PCA to data/fitted_pca.joblib for next time!')
