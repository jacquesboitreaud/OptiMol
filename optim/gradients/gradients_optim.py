# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:45:02 2019

@author: jacqu

Gradient descent optimization of objective target function in latent space.

Starts from 1 seed compound or random point in latent space (sampled from prior N(0,1))

TODO : 
    - add tanimoto similarity to seed compound
"""

import os
import sys
import argparse
import pickle
import torch
import numpy as np
from numpy.linalg import norm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw, QED, Crippen, Descriptors
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt
from selfies import decoder

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.rdkit_to_nx import smiles_to_nx
from dataloaders.molDataset import molDataset
from utils import *
from model import Model, model_from_json

parser = argparse.ArgumentParser()

parser.add_argument('--name', help="Saved model directory, in /results/saved_models",
                    default='inference_default')

parser.add_argument('-r', "--random", help="Start from random latent point ", action='store_true',
                    default=True)

parser.add_argument('-s', "--seed", help="seed smiles to optimize ", type=str,
                    default='O=C(NC1=CCCC1=O)NC1=CCN(c2ncnc3ccccc23)CC1')

parser.add_argument("--steps", help="Number of gradients ascent steps", type=int,
                    default=100)
parser.add_argument("--lr", help="Learning rate at each step", type=float,
                    default=1e-1)  # Best value still to find. Need experiments.

parser.add_argument('-v', "--verbose", help="Extensive step by step logs ", action='store_true',
                    default=False)

args, _ = parser.parse_known_args()

# =======================================

early_stopping_QED = 0.940


def eps(props, aff=None):
    """ 
    Objective function to maximize. 
    props =  array [QED, logP, molWt], shape (N_properties,)
    aff = array with affinities, shape (N_targets,) 
    """
    obj_expression = 'QED(m)'
    if type(props) == str:
        return obj_expression  # return function expression

    QED, logP, molWt = props
    # return  (QED-1)**2 + aff[0]

    obj = QED

    return obj


molecules = []
data = molDataset(csv_path=None,
                  maps_path='../map_files/',
                  vocab='selfies',
                  build_alphabet=False,
                  props=['QED', 'logP', 'molWt'],
                  targets=[])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model_from_json(args.name)
model.to(device)
model.eval()

print(f'>>> Gradient ascent to maximize {eps("")}. {args.steps} iterations. ')

# ================== Starting from a seed compound ======================
if not args.random:
    # Begin optimization with seed compound : a DUD decoy   
    s_seed = 'O=C(NC1=CCCC1=O)NC1=CCN(c2ncnc3ccccc23)CC1'
    m0 = Chem.MolFromSmiles(s_seed)
    fp0 = MACCSkeys.GenMACCSKeys(m0)  # fingerprint (MACCS)

    print(">>> Starting from seed molecule")
    Draw.MolToMPL(m0, size=(120, 120))
    plt.show(block=False)
    plt.pause(0.1)

    # Pass to loader 
    data.pass_smiles_list([s_seed])
    graph, _, _, _ = data.__getitem__(0)
    send_graph_to_device(graph, model.device)

    # pass to model
    z = model.encode(graph, mean_only=True)
    z = z.unsqueeze(dim=0)
else:
    # ======================= Starting from random point ===============
    print(">>> Starting from random latent vector")
    z = model.sample_z_prior(1)
    z.requires_grad = True

# ================= Optimization process ================================

lr = args.lr
ctr_valid = 0
valid_smiles = []
best_observed = 0

for i in range(args.steps):
    # Objective function 
    epsilon = eps(model.props(z)[0], model.affs(z)[0])
    g = torch.autograd.grad(epsilon, z)
    with torch.no_grad():  # Gradient descent, don't track
        z = z + lr * g[0]
        if i % 20 == 0:
            lr = 0.9 * lr

        out = model.decode(z)
        smi = model.probas_to_smiles(out)[0]
        if data.language == 'selfies':
            smi = decoder(smi)

        # for beam search decoding
        # out=model.decode_beam(z)
        # smi = model.beam_out_to_smiles(out)[0]

        m = Chem.MolFromSmiles(smi)
        pred_qed = model.props(z)[0, 0].item()

        if m != None:
            ctr_valid += 1
            logP = Chem.Crippen.MolLogP(m)
            qed = Chem.QED.default(m)
            valid_smiles.append((i, smi, pred_qed, qed))  # tuples (smiles, step)

            if qed > best_observed:
                best_observed = qed
                best_compound = smi
                best_step = i

            if args.verbose:
                Draw.MolToMPL(m, size=(120, 120))
                plt.show(block=False)
                plt.pause(0.1)

                print(f'predicted logP: {model.props(z)[0, 1].item():.2f}, true: {logP:.2f}')
                print(f'predicted QED: {model.props(z)[0, 0].item():.4f}, true: {qed:.4f}')
                # print(f'predicted aff: {model.affs(z)[0,0].item():.2f}')

            if qed >= early_stopping_QED:
                print(f'-> Early stopping QED value reached at step {i}')
                break

        elif args.verbose:
            print('invalid smiles')
    z.requires_grad = True

# Display results after iterations : 

m = Chem.MolFromSmiles(best_compound)

print('Best compound smiles :', best_compound)
print('True QED :', best_observed)
print('Step observed :', best_step)
print(f'{ctr_valid}/{args.steps} steps with valid smiles')

if args.verbose:
    for tup in valid_smiles:
        print(f'step {tup[0]}: {tup[1]}, pred_qed = {tup[2]}, true = {tup[3]}')

if not args.random:
    fp = MACCSkeys.GenMACCSKeys(m)
    tanimoto = DataStructs.FingerprintSimilarity(fp0, fp, metric=DataStructs.TanimotoSimilarity)
    print('Tanimoto similarity to seed compound: ', tanimoto)
