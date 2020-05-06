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

from utils import *
from model import model_from_json
from oracles import qed, deterministic_cdf_oracle, normal_cdf_oracle
from gen_train import GenTrain
from gen_prob import GenProb

from docking import dock, set_path


parser = argparse.ArgumentParser()

parser.add_argument('--prior_name', type=str, default='kekule') # the prior VAE (pretrained)
parser.add_argument('--search_name', type=str, default='search_vae') # the prior VAE (pretrained)

parser.add_argument('--oracle', type=str, default='qed') # qed for toy oracle, 'aff' for docking 
parser.add_argument('--computer', type=str, default='rup') # Computer to use for docking 

parser.add_argument('--iters', type=int, default=50) # Number of iterations
parser.add_argument('--Q', type=float, default=0.5) # quantile of scores accepted
parser.add_argument('--M', type=float, default=1000) # Nbr of samples at each iter 

# =======

args = parser.parse_args()

# Initialization 
t=1

# Load or train prior VAE 

prior_model = model_from_json(args.prior_name)

# Initialize search vae q
search_model = model_from_json(args.prior_name)

# Docking params 
if args.oracle == 'aff' : 
    print(f'Docking params setup for {args.computer}')
    pythonsh, vina = set_path(args.computer)

for t in range(args.iters):
    
    print(f'> start iteration {t}')
    
    # Sampling from q
    samples = search_model.sample_z_prior(n_mols = args.M)
    gen_seq = search_model.decode(samples)
    samples = search_model.probas_to_smiles(gen_seq)
    samples= [decoder(s) for s in samples]
    
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
    weights = 1- deterministic_cdf_oracle(scores, gamma)
    
    
    # Update search model 
    GenTrain(samples, weights, model = search_model, savepath = os.path.join(script_dir,'results/saved_models',args.search_name) )
    
    # Load new search model_weights 
    search_model.load_state_dict(torch.load(os.path.join(script_dir,'results/saved_models',args.search_name, 'weights.pth')))
    
    # Get some prints and repeat 
    

