# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Evaluate training of Graph2Smiles VAE (RGCN encoder, GRU decoder, beam search decoding). 


"""

import os
import sys

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import sys
import torch
import dgl

from rdkit.Chem import Draw

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from joblib import dump, load
from sklearn.decomposition import PCA

from dataloaders.molDataset import molDataset, Loader
from data_processing.rdkit_to_nx import smiles_to_nx
from model import Model, Loss, RecLoss
from eval.eval_utils import *
from utils import *
from plot_tools import *

# Choose model to load :
batch_size = 100
model_path = f'saved_models/baseline.pth'

# Should be same as for training...
properties = ['QED', 'logP', 'molWt']
targets = ['aa2ar', 'drd3']

# Load eval set
loaders = Loader(csv_path='data/moses_train.csv',
                 n_mols=500000,
                 num_workers=0,
                 batch_size=batch_size,
                 props=properties,
                 targets=targets,
                 test_only=True,
                 shuffle=True)
rem, ram, rchim, rcham = loaders.get_reverse_maps()

_, _, test_loader = loaders.get_data()

# Model & hparams
device = 'cuda' if torch.cuda.is_available() else 'cpu'

params = pickle.load(open('saved_models/params.pickle', 'rb'))
model = Model(**params).to(device)
model.load_state_dict(torch.load(model_path))

# Print model summary
print(model)
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
            print(batch_idx)

        graph = send_graph_to_device(graph, device)

        # Latent embeddings
        z = model.encode(graph, mean_only=True)  # z_shape = N * l_size
        z = z.cpu().numpy()
        z_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = z

    # ===================================================================
    # PCA fitting
    # ===================================================================
    fitted_pca = fit_pca(z)
    dump(fitted_pca, 'eval/fitted_pca.joblib')
    print('Fitted and saved PCA for next time!')
