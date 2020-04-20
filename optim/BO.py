# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Optimize affinity with bayesian optimization 

"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd 

import pickle
import torch.utils.data

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from selfies import decoder

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler



if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    repo_dir = os.path.join(script_dir, '..')
    sys.path.append(repo_dir)
    sys.path.append(os.path.join(repo_dir,'..', 'vina_docking')) # path to vina_docking repo with scoring func 
    
    from dataloaders.molDataset import Loader
    from model import Model
    from utils import *
    from BO_utils import get_fitted_model
    from score_function import score
    from score_function_batch import dock_batch

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--n_steps", help="Nbr of optim steps", type=int, default=50)
    parser.add_argument('-m', '--model', help="saved model weights fname. Located in saved_model_w subdir",
                        default='saved_model_w/baseline.pth')
    parser.add_argument('-v', '--vocab', default='selfies') # vocab used by model 
    args = parser.parse_args()

    # ==============
    
    # Loader for initial sample
    loader = Loader(props=[], 
                    targets=[], 
                    csv_path = None,
                    vocab=args.vocab, 
                    num_workers = 0)

    # Load model (on gpu if available)
    params = pickle.load(open(os.path.join(repo_dir,'saved_model_w/model_params.pickle'), 'rb'))  # model hparams
    model = Model(**params)
    model.load(os.path.join(repo_dir,args.model))
    model.eval()
    
    d = 64
    dtype = torch.float
    device = 'cpu'
    bounds = torch.tensor([[-4.0] * d, [4.0] * d], device=device, dtype=dtype)
    BO_BATCH_SIZE = 3
    N_STEPS = args.n_steps
    MC_SAMPLES = 2000
    
    
    seed=1
    torch.manual_seed(seed)
    best_observed = []
    state_dict = None
    
    # Generate initial data 
    df = pd.read_csv(os.path.join(repo_dir,'data','drd3_1k_samples.csv'))
    scores_init = df.scores
    loader.graph_only=False
    train_x = model.embed( loader, df) # z has shape (N_molecules, latent_size)
    train_obj = torch.tensor(scores_init).view(-1,1)
    best_value = max(scores_init)
    best_observed.append(best_value)
    
    # Acquisition function 
    def optimize_acqf_and_get_observation(acq_func, device):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
        
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(d, dtype=dtype, device=device), 
                torch.ones(d, dtype=dtype, device=device),
            ]),
            q=BO_BATCH_SIZE,
            num_restarts=10,
            raw_samples=200,
        )
    
        # observe new values 
        new_z = unnormalize(candidates.detach(), bounds=bounds)
        
        # Decode z into smiles
        gen_seq = model.decode(new_z)
        smiles = model.probas_to_smiles(gen_seq)
        if(args.vocab=='selfies'):
            smiles =[ decoder(s) for s in smiles]
        
        if BO_BATCH_SIZE > 1 :
            new_scores = dock_batch(smiles).unsqueeze(-1)  # add output dimension
            return smiles, new_z, new_scores
        else:
            new_score = score(smiles).unsqueeze(-1)
            return smiles, new_z, new_score
    
    
    
    # ========================================================================
    # run N_BATCH rounds of BayesOpt after the initial random batch
    # ========================================================================
    
    for iteration in range(N_STEPS):    
    
        # fit the model
        model = get_fitted_model(
            normalize(train_x, bounds=bounds), 
            standardize(train_obj), 
            state_dict=state_dict,
        )
        
        # define the qNEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
        qEI = qExpectedImprovement(model=model, sampler=qmc_sampler, best_f=standardize(train_obj).max())
    
        # optimize and get new observation
        new_smiles, new_z, new_score = optimize_acqf_and_get_observation(qEI)
        
        # save acquired scores for next time 
        print(new_smiles)
        print(new_score)
    
        # update training points
        train_x = torch.cat((train_x, new_z))
        train_obj = torch.cat((train_obj, new_score))
    
        # update progress
        best_value = train_obj.max().item()
        best_observed.append(best_value)
        
        state_dict = model.state_dict()
        
        print(".", end='')
