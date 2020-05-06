# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:08:41 2020

@author: jacqu


Proba to generate a sample given a model 
"""

from torch import nn 
import torch

def GenProb(x,z, model):
    """
    Inputs :
        x : array indices. Shape batch_size * seq_len
        z : latent point 
        model : vae (requires .decode function)
    """
    
    out = model.decode(z)
    logprob = nn.LogSoftmax(out, dim = 1) # shape is (N, num_chars_in_alphbabet, sequence_len)
    
    # logprob of x 
    one_hot = torch.nn.functional.one_hot(x, model.num_chars)
    one_hot = one_hot.transpose(2,1) # to shape N, n_chars, sequence_len
    
    logprob = logprob * one_hot
    logprob_x = torch.sum(logprob.reshape(z.shape[0], -1), dim = 1)
    
    return logprob_x 

