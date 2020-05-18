# -*- coding: utf-8 -*-
"""
Created on Wed May  6 17:08:41 2020

@author: jacqu


Proba to generate a sample given a model 
"""

import torch
from torch import nn


def GenProb(x, z, model):
    """
    Inputs :
        x : array indices. Shape batch_size * seq_len
        z : latent point 
        model : vae (requires .decode function)
    """
    with torch.no_grad():
        out = model.decode(z)
        sigma = nn.LogSoftmax(dim=1)  # shape is (N, num_chars_in_alphbabet, sequence_len)
        logprob = sigma(out)

        # logprob of x 
        one_hot = torch.nn.functional.one_hot(x, model.voc_size)
        one_hot = one_hot.transpose(2, 1).float()  # to shape N, n_chars, sequence_len

        logprob = logprob * one_hot
        logprob_x = torch.sum(logprob.reshape(z.shape[0], -1), dim=1)

    return logprob_x.cpu()


if __name__ == '__main__':
    from utils import *
    from dgl_utils import send_graph_to_device
    from model import model_from_json
    import numpy as np

    print('Testing for a random batch of 12 molecules')

    model = model_from_json('kekule')

    x = np.random.randint(0, 33, size=(12, 54))
    x = torch.tensor(x, dtype=torch.long)

    z = model.sample_z_prior(n_mols=12)

    true_dec = model.decode(z)
    _, true_dec = torch.max(true_dec, dim=1)

    l_true = GenProb(true_dec, z, model)
    l = GenProb(x, z, model)

    print('logprob of the true decoded x |z ', l_true.cpu().detach())
    print('logprob of a randomly sampled x |z ', l.cpu().detach())
