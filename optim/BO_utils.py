# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Optimize affinity with bayesian optimization 

"""
import os, sys 
import numpy as np

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_model




def gen_initial_data(scores_csv_file):
    # generate training data  
    # train_x = N * 64 
    # train_obj = N*1 
    # Best observed value is a scalar 
    
    train_x = unnormalize(torch.rand(n, d, device=device, dtype=dtype), bounds=bounds)
    
    train_obj = score(decode(train_x)).unsqueeze(-1)  # add output dimension
    
    best_observed_value = train_obj.max().item()
    return train_x, train_obj, best_observed_value

def get_fitted_model(train_x, train_obj, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll.to(train_x)
    fit_gpytorch_model(mll)
    return model



