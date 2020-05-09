# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:10:04 2020

@author: jacqu

CbAS iterative procedure 

"""
import os, sys 

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(script_dir, '..','dataloaders'))
sys.path.append(os.path.join(script_dir, '..','docking'))

import torch
import numpy as np 
import argparse 

from selfies import decoder
from rdkit import Chem

from utils import *
from dgl_utils import * 
from model import model_from_json
from oracles import qed, deterministic_cdf_oracle, normal_cdf_oracle
from gen_train import GenTrain
from gen_prob import GenProb

from docking import dock, set_path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--prior_name', type=str, default='inference_default') # the prior VAE (pretrained)
    parser.add_argument('--search_name', type=str, default='search_vae') # the prior VAE (pretrained)
    
    parser.add_argument('--oracle', type=str, default='qed') # qed for toy oracle, 'aff' for docking 
    parser.add_argument('--computer', type=str, default='rup') # Computer to use for docking 
    
    parser.add_argument('--iters', type=int, default=50) # Number of iterations
    parser.add_argument('--Q', type=float, default=0.8) # quantile of scores accepted

    parser.add_argument('--M', type=int, default=10000) # Nbr of samples at each iter 
    
    # Params of the search-model finetuning (seems sensitive)
    parser.add_argument('--epochs', type=int, default=10) # Number of iterations
    parser.add_argument('--learning_rate', type=float, default=1e-4) # Number of iterations
    parser.add_argument('--clip_grad_norm', type=float, default=5.0) # quantile of scores accepted
    
    # =======
    
    args = parser.parse_args()
    
    # Initialization 
    
    # Load or train prior VAE 
    
    prior_model = model_from_json(args.prior_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prior_model.to(device)
    
    # Initialize search vae q
    savepath = os.path.join(script_dir,'results/saved_models',args.search_name)
    searchTrainer = GenTrain(args.prior_name, savepath, epochs = args.epochs, device=device, 
                             lr = args.learning_rate, clip_grad = args.clip_grad_norm)
    
    # Docking params 
    if args.oracle == 'aff' : 
        print(f'Docking params setup for {args.computer}')
        pythonsh, vina = set_path(args.computer)
    
    for t in range(1, args.iters+1):
        
        print(f'> start iteration {t}')
        
        # Sampling from q (split into batches of size 100 )
        samples_z = searchTrainer.model.sample_z_prior(n_mols = args.M)
        sample_selfies = []
        weights = []
        
        for batch_idx in range(args.M//100):
            batch_z = samples_z[batch_idx*100:(batch_idx+1)*100]
            gen_seq = searchTrainer.model.decode(batch_z)
            _ , sample_indices = torch.max(gen_seq, dim=1)
            sample_selfies += searchTrainer.model.indices_to_smiles(sample_indices)
            
            # Compute weights while we have indices and store them: p(x|z, theta)/p(x|z, phi)
            weights.append(GenProb(sample_indices, batch_z, prior_model) / GenProb(sample_indices, batch_z, searchTrainer.model))
            
        samples= [decoder(s) for s in sample_selfies]
        print(samples[:10]) # debugging 
        weights = torch.cat(weights, dim = 0)
        
        # scoring 
        if args.oracle == 'aff':
            scores = [dock(s,i, pythonsh,vina) for i,s in enumerate(samples)]
        elif args.oracle == 'qed' : #toy oracle 
            scores = qed(samples) # function takes a list of mols 
        
        # Sort scores and find Qth quantile 
        sorted_sc = sorted(scores)
        gamma = np.quantile(sorted_sc, args.Q)
        print(f"step {t}/{args.iters}, gamma = {gamma}")
        
        # Weight samples 
        scores = np.array(scores)
        
        # Update weights by proba that oracle passes threshold
        weights = weights *(1- deterministic_cdf_oracle(scores, gamma)) # weight 0 if oracle < gamma
        
        # Drop invalid and correct smiles to kekule format to avoid reencoding issues when training search model 
        good_indices =  []
        for i,s in enumerate(samples):
            m=Chem.MolFromSmiles(s)
            if m!=None and weights[i]>0: # get rid of all samples with weight 0 (do not count in CbAS loss)
                good_indices.append(i)
                
        samples = [samples[i] for i in good_indices]
        weights = weights[good_indices]
                
        print(f'{len(good_indices)}/{args.M} samples kept')
        
        
        # Update search model 
        searchTrainer.step('smiles', samples, weights)

        # Get some prints and repeat 
    

