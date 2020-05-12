import subprocess

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:10:04 2020

@author: jacqu

CbAS iterative procedure 

"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

import torch
import numpy as np
import argparse

from selfies import decoder
from rdkit import Chem

from utils import *
from dgl_utils import *
from model import model_from_json
from cbas.oracles import qed, deterministic_cdf_oracle, normal_cdf_oracle
from cbas.gen_train import GenTrain
from cbas.gen_prob import GenProb
from docking.docking import dock, set_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--search_name', type=str, default='search_vae')  # the prior VAE (pretrained)

    parser.add_argument('--oracle', type=str, default='qed')  # qed for toy oracle, 'aff' for docking
    parser.add_argument('--computer', type=str, default='rup')  # Computer to use for docking

    parser.add_argument('--procs', type=int, default=0)  # Number of processes for VAE dataloading

    parser.add_argument('--iters', type=int, default=25)  # Number of iterations
    parser.add_argument('--Q', type=float, default=0.6)  # quantile of scores accepted

    parser.add_argument('--M', type=int, default=1000)  # Nbr of samples at each iter

    # Params of the search-model finetuning (seems sensitive)
    parser.add_argument('--epochs', type=int, default=5)  # Number of iterations
    parser.add_argument('--learning_rate', type=float, default=1e-4)  # Number of iterations
    parser.add_argument('--beta', type=float, default=0.2)  # KL weight in loss function
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)  # quantile of scores accepted

    # =======

    args = parser.parse_args()

    # Initialization

    # Load or train prior VAE

    prior_model = model_from_json(args.prior_name)
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'
    prior_model.to(device)

    # Initialize search vae q
    savepath = os.path.join(script_dir, 'results/saved_models', args.search_name)
    searchTrainer = GenTrain(args.prior_name, savepath, epochs=args.epochs, device=device,
                             lr=args.learning_rate, clip_grad=args.clip_grad_norm, beta=args.beta,
                             processes=args.procs, DEBUG=True)

    # Docking params
    if args.oracle == 'aff':
        print(f'Docking params setup for {args.computer}')
        pythonsh, vina = set_path(args.computer)

    for t in range(1, args.iters + 1):

        print(f'> start iteration {t}')


        # Sampling from q (split into batches of size 100 )
        def get_samples(prior_model, searchTrainer):
            sample_selfies = []
            weights = []
            sample_selfies_set = set()
            tries = 0
            stop = 100
            batch_size = 100

            # Importance weights
            while tries < stop or len(sample_selfies) < args.M:
                new_ones = 0

                # Get raw samples
                samples_z = searchTrainer.model.sample_z_prior(n_mols=batch_size)
                gen_seq = searchTrainer.model.decode(samples_z)
                _, sample_indices = torch.max(gen_seq, dim=1)

                # Compute weights while we have indices and store them: p(x|z, theta)/p(x|z, phi)
                batch_weights = GenProb(sample_indices, samples_z, prior_model) / \
                                GenProb(sample_indices, samples_z, searchTrainer.model)

                # Check the novelty
                batch_selfies = searchTrainer.model.indices_to_smiles(sample_indices)
                for i, s in enumerate(batch_selfies):
                    new_selfie = decoder(s)
                    if new_selfie not in sample_selfies_set:
                        new_ones += 1
                        sample_selfies_set.add(new_selfie)
                        sample_selfies.append(new_selfie)
                        weights.append(batch_weights[i])
                tries += 1

                print(f'{new_ones}/{batch_size} unique smiles sampled')
                print(samples[:10])  # debugging
                weights = torch.cat(weights, dim=0)
            return samples, weights


        # TODO: package as a slurm standalone

        # SIMULATE DOCKING
        slurm_docker_path = os.path.join(script_dir, 'slurm_docker.sh')
        cmd = f'sbatch {slurm_docker_path}'
        a = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        id_run = a.split()[3]
        # print(f'launched docking run with id {id_run}')

        """
        # WAIT FOR COMPLETION BEFORE RUNNING THAT
        cmd_2 = f"sbatch --depend=afterany:{id_run} arrayer.sh"
        print(cmd_2)
        out_2 = subprocess.run(cmd_2.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        id_run_2 = out_2.split()[3]
        print(f'launched second run with id {id_run_2}')
        """
        # scoring
        if args.oracle == 'aff':
            scores = [dock(s, i, pythonsh, vina) for i, s in enumerate(samples)]
        elif args.oracle == 'qed':  # toy oracle
            scores = qed(samples)  # function takes a list of mols

        # TODO : gather docking with the main, sort scores and get quantiles, load old model and importance
        # TODO : sampling weights, do one training and save this model.
        # TODO : Then package as a standalone slurm call
        # Sort scores and find Qth quantile
        sorted_sc = sorted(scores)
        gamma = np.quantile(sorted_sc, args.Q)
        print(f"step {t}/{args.iters}, gamma = {gamma}")

        # Weight samples
        scores = np.array(scores)

        # Update weights by proba that oracle passes threshold
        weights = weights * (1 - deterministic_cdf_oracle(scores, gamma))  # weight 0 if oracle < gamma

        # Drop invalid and correct smiles to kekule format to avoid reencoding issues when training search model
        good_indices = []
        for i, s in enumerate(samples):
            m = Chem.MolFromSmiles(s)
            if m is not None and weights[i] > 0:  # get rid of all samples with weight 0 (do not count in CbAS loss)
                good_indices.append(i)

        samples = [samples[i] for i in good_indices]
        weights = weights[good_indices]

        print(f'{len(good_indices)}/{args.M} samples kept')

        # Update search model
        searchTrainer.step('smiles', samples, weights)

        # Get some prints and repeat
