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

from model import Model, model_from_json
from oracles import qed
from gen_train import GenTrain
from gen_prob import GenProb



parser = argparse.ArgumentParser()

parser.add_argument('--prior_name', type=str, default='default_prior') # the prior VAE (pretrained)


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

for t in range(args.iters):
    
    print(f'> start iteration {t}')
    
    # Sampling from q
    samples = search_model.sample_z_prior(n_mols = args.M)
    
    # scoring 
    if args.oracle == 'aff':
        scores = [dock(s) for s in samples]
    elif args.oracle == 'qed' : #toy oracle 
        scores = qed(samples) # function takes a list of mols 
    
    # Sort scores and find Qth quantile 
    sorted_sc = ...
    
    # Weight samples 
    #TODO
    
    
    # Update search model 
    GenTrain(samples, weights)
    
    # Get some prints and repeat 
    

