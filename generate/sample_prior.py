# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space // random samples from diagonal normal dist 
Run from repo root. 

"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import argparse
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F
from selfies import decoder
from rdkit import Chem

import seaborn as sns
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    
    from dataloaders.molDataset import molDataset, Loader
    from data_processing.rdkit_to_nx import smiles_to_nx
    from model import Model, model_from_json
    from utils import *

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="Saved model directory, in /results/saved_models",
                        default='finetuned_10')
    
    parser.add_argument('-N', "--n_mols", help="Nbr to generate", type=int, default=5000)
    parser.add_argument('-v', '--vocab', default='selfies')  # vocab used by model

    parser.add_argument('-o', '--output_file', type=str, default='data/gen.txt')
    parser.add_argument('-b', '--use_beam', action='store_true', help="use beam search (slow!)")
    
    parser.add_argument( '--qed', action='store_true', help="plot qed distrib")

    args, _ = parser.parse_known_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ==============

    # Load model (on gpu if available)
    model = model_from_json(args.name)
    model.to(device)

    compounds = []
    cpt = 0 
    
    with torch.no_grad():
        batch_size = min(args.n_mols, 100)
        n_batches = int(args.n_mols / batch_size) + 1
        print(f'>>> Sampling {args.n_mols} molecules from prior distribution in latent space')

        for b in range(n_batches):

            z = model.sample_z_prior(batch_size)
            gen_seq = model.decode(z)

            # Sequence of ints to smiles 
            if not args.use_beam :
                selfies = model.probas_to_smiles(gen_seq)
            else:
                selfies = model.beam_out_to_smiles(gen_seq)

            if args.vocab == 'selfies' :
                smiles = [decoder(s, bilocal_ring_function=True) for s in selfies]

            compounds += smiles
            mols = [Chem.MolFromSmiles(s) for s in smiles]
            for i,m in enumerate(mols) :
                if m==None:
                    print(smiles[i], ' , invalid, selfies output was : ')
                    print(selfies[i])
                else:
                    cpt +=1

    Ntot = len(compounds)
    unique = list(np.unique(compounds))
    N = len(unique)
    
    print(100*cpt/Ntot, '% valid molecules' )
    print(100*N/Ntot, '% unique molecules' )
    
    if args.qed :
        qed = [Chem.QED.qed(m) for m in mols]
        sns.distplot(qed, norm_hist = True)
        plt.title('QED distrib of samples')

    out = os.path.join(script_dir, '..', args.output_file)
    with open(out, 'w') as f:
        for s in unique:
            f.write(s)
            f.write('\n')
    print(f'wrote {N} unique compounds / {Ntot} to {args.output_file}')
