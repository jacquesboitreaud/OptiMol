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
from rdkit import Chem

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
    from model_baseline import Model
    from utils import *
    from BO_utils import get_fitted_model
    from score_function import score
    from score_function_batch import dock_batch

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', "--n_steps", help="Nbr of optim steps", type=int, default=50)
    parser.add_argument('-m', '--model', help="saved model weights fname. Located in saved_models subdir",
                        default='saved_models/baseline.pth')
    parser.add_argument('-v', '--vocab', default='selfies') # vocab used by model 
    
    parser.add_argument('-d', '--device', default='cuda') # 'cpu or 'cuda'. 
    parser.add_argument('-o', '--objective', default='qed') # 'qed' or 'aff'
    args = parser.parse_args()

    # ==============
    
    # Loader for initial sample
    loader = Loader(props=[], 
                    targets=[], 
                    csv_path = None,
                    vocab=args.vocab, 
                    num_workers = 0)

    # Load model (on gpu if available)
    params = pickle.load(open(os.path.join(repo_dir,'saved_models/model_params.pickle'), 'rb'))  # model hparams
    model = Model(**params)
    model.load(os.path.join(repo_dir,args.model))
    model.eval()
    
    d = 64
    dtype = torch.float
    device = args.device
    # make sure model settings are coherent 
    model.device = device 
    model.to(device)
    bounds = torch.tensor([[-4.0] * d, [4.0] * d], device=device, dtype=dtype)
    BO_BATCH_SIZE = 10
    N_STEPS = args.n_steps
    MC_SAMPLES = 2000
    
    
    seed=1
    torch.manual_seed(seed)
    best_observed = []
    state_dict = None
    
    # Generate initial data 
    df = pd.read_csv(os.path.join(repo_dir,'data','1k_sample.csv'))
    if args.objective == 'aff':
        scores_init = df.scores
    else:
        scores_init = [Chem.QED.qed(Chem.MolFromSmiles(s)) for s in df.smiles]
        
    loader.graph_only=False
    train_z = torch.tensor(model.embed( loader, df)).to(device) # z has shape (N_molecules, latent_size)
    train_obj = torch.tensor(scores_init).view(-1,1).to(device)
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
            
            if args.objective == 'aff':
                new_scores, _ = dock_batch(smiles)
                new_scores = torch.tensor(new_scores)
            elif args.objective =='qed':
                mols = [Chem.MolFromSmiles(s) for s in smiles ]
                new_scores = torch.zeros(len(smiles), dtype = torch.float)
                for i,m in enumerate(mols):
                    if m!=None:
                        new_scores[i]=Chem.QED.qed(m)
                
            new_scores = new_scores.unsqueeze(-1)  # add output dimension
            return smiles, new_z, new_scores
        
        else:
            if args.objective == 'aff':
                new_score, _ = score(smiles)
            elif args.objective == 'qed':
                new_score = Chem.QED.qed(Chem.MolFromSmiles(smiles))
                
            new_score = torch.tensor(new_score).unsqueeze(-1)
            return smiles, new_z, new_score
    
    
    
    # ========================================================================
    # run N_BATCH rounds of BayesOpt after the initial random batch
    # ========================================================================
    
    for iteration in range(N_STEPS):    
    
        # fit the model
        GP_model = get_fitted_model(
            normalize(train_z, bounds=bounds), 
            standardize(train_obj), 
            state_dict=state_dict,
        )
        
        # define the qNEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
        qEI = qExpectedImprovement(model=GP_model, sampler=qmc_sampler, best_f=standardize(train_obj).max())
    
        # optimize and get new observation
        new_smiles, new_z, new_score = optimize_acqf_and_get_observation(qEI, device)
        
        # save acquired scores for next time 
        print(new_smiles)
        print(new_score)
    
        # update training points
        new_z.to(device)
        new_score = new_score.to(device)
        
        train_z = torch.cat((train_z, new_z), dim=0)
        train_obj = torch.cat((train_obj, new_score), dim=0)
    
        # update progress
        best_value = train_obj.max().item()
        best_observed.append(best_value)
        
        state_dict = GP_model.state_dict()
        
        print(".", end='')
