# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space // random samples from diagonal normal dist 
Run from repo root. 

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

from selfies import decoder

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, 'dataloaders'))
    sys.path.append(os.path.join(script_dir, 'data_processing'))
    
    from dataloaders.molDataset import molDataset
    from model_noTF import Model
    from utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--n_mols", help="Nbr to generate", type=int, default=1000)
    parser.add_argument('-m', '--model', help="saved model weights fname. Located in saved_model_w subdir",
                        default='saved_model_w/autoreg_model_iter_360000.pth')
    parser.add_argument('-v', '--vocab', default='selfies') # vocab used by model 
    
    parser.add_argument('-o', '--output_file', type=str, default='data/gen_a.txt')
    parser.add_argument('-b', '--use_beam', action='store_true', help="use beam search (slow!)")
    

    args = parser.parse_args()

    # ==============

    # Load model (on gpu if available)
    params = pickle.load(open(os.path.join(script_dir,'..','saved_model_w/model_params.pickle'), 'rb'))  # model hparams
    model = Model(**params)
    model.load(os.path.join(script_dir, '..', args.model))

    model.eval()

    compounds = []

    with torch.no_grad():
        batch_size = min(args.n_mols, 100)
        n_batches = int(args.n_mols/batch_size)+1
        print(f'>>> Sampling {args.n_mols} molecules from prior distribution in latent space')
        
        for b in range(n_batches):
            
            z = model.sample_z_prior(batch_size)
            gen_seq = model.decode(z)
            
            # Sequence of ints to smiles 
            if (not args.use_beam):
                smiles = model.probas_to_smiles(gen_seq)
            else:
                smiles = model.beam_out_to_smiles(gen_seq)
            
            if(args.vocab=='selfies'):
                smiles =[ decoder(s) for s in smiles]
                
            compounds += smiles
    
    Ntot = len(compounds)
    unique = list(np.unique(compounds))
    N = len(unique)
    
    out = os.path.join(script_dir,'..',args.output_file)
    with open(out, 'w') as f:
        for s in unique:
            f.write(s)
            f.write('\n')
    print(f'wrote {N} unique compounds / {Ntot} to {args.output_file}')
