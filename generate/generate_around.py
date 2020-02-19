# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space // around a given seed molecule

"""
import os
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import argparse
import sys
import torch
import numpy as np

import pickle
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if __name__ == "__main__":

    from dataloaders.molDataset import molDataset
    from model import Model
    from utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--seed', help="seed SMILES", type=str,
                        default='O=C(O)CCCc1ccc(N(CCF)CCCl)cc1')
    parser.add_argument('-d', "--distance", help="Euclidian distance to seed mean", type=int, default=5)
    parser.add_argument('-n', "--n_mols", help="Nbr to generate", type=int, default=100)
    parser.add_argument('-m', '--model', help="saved model weights fname. Located in saved_model_w subdir",
                        default='baseline.pth')
    parser.add_argument('-o', '--output_file', type=str, default='gen_batch.txt')
    parser.add_argument('-b', '--use_beam', action='store_true', help="use beam search")
    

    args = parser.parse_args()

    # ==============

    model_path = f'../saved_model_w/{args.model}'
    # Load model (on gpu if available)
    params = pickle.load(open('../saved_model_w/params.pickle', 'rb'))  # model hparams
    model = Model(**params)
    model.load(model_path)
    # Provide the model with characters corresponding to indices, for smiles generation 
    model.set_smiles_chars(char_file="../map_files/zinc_chars.json")
    model.eval()

    compounds = []

    data = molDataset(maps_path='../map_files/',
                      csv_path=None)
    data.pass_smiles_list([args.seed])
    g_dgl, _, _, _ = data.__getitem__(0)

    with torch.no_grad():
        send_graph_to_device(g_dgl, model.device)
        gen_seq, _, _ = model.sample_around_mol(g_dgl, dist=args.distance, beam_search=args.use_beam,
                                                attempts=args.n_mols, props=False,
                                                aff=False)  # props & affs returned in _

    # Sequence to smiles 
    if (not args.use_beam):
        smiles = model.probas_to_smiles(gen_seq)
    else:
        smiles = model.beam_out_to_smiles(gen_seq)

    compounds += smiles

    unique = list(np.unique(compounds))

    with open(args.output_file, 'w') as f:
        for s in unique:
            f.write(s)
            f.write('\n')
    print(f'wrote output batch to {args.output_file}')
