# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

Loss functions for VAE, multitask & contrastive. 

"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================= Loss functions ====================================

def VAELoss(out, indices, mu, logvar):
    """ 
    plain VAE loss. 
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # returns zeros for multitask loss terms
    return CE, KL


def weightedPropsLoss(p_target, p_pred, weights):
    """
    Weighted loss for chemical properties. N_properties = p_target.shape[1]. 
    weights should be a FloatTensor of shape N_properties, indicates weight of each prop. 
    Adjust props weights according to absolute values of properties (molWt ~ 10**2 QED for example)
    """
    mse = nn.MSELoss(reduction='mean')

    loss = weights[0] * mse(p_pred[:, 0], p_target[:, 0])

    for i in range(1, p_target.shape[1]):
        loss += weights[i] * mse(p_pred[:, i], p_target[:, i])

    return loss


def affsClassifLoss(a_target, a_pred, classes_weights):
    """ 
    NLL classification loss applied to logSoftmax of affinity bin prediction. Bins weighted with classes_weights. 
    """
    a_target = a_target.squeeze()
    aff_loss = F.nll_loss(a_pred, target=a_target, weight=classes_weights)  # class zero does not contribute to loss

    return aff_loss


def affsRegLoss(a_target, a_pred, weight, ignore=[-9., -7.]):
    """
    Regression MSE loss for affinity values outside the 'ignore' interval. 
    """
    mse = nn.MSELoss()

    aff_loss = torch.tensor(0.0).to('cuda')
    for i in range(a_target.shape[0]):
        if a_target[i] < 0:  # Affinity score available
            if a_target[i] < ignore[0] or a_target[i] > ignore[1]:
                aff_loss += mse(a_pred[i], a_target[i])

    return aff_loss * weight


def tripletLoss(z_i, z_j, z_l, margin=2):
    """ For disentangling by separating known actives in latent space """
    dij = torch.norm(z_i - z_j, p=2,
                     dim=1)  # z vectors are (N*l_size), compute norm along latent size, for each batch item.
    dil = torch.norm(z_i - z_l, p=2, dim=1)
    loss = torch.max(torch.cuda.FloatTensor(z_i.shape[0]).fill_(0), dij - dil + margin)
    # Embeddings distance loss 
    return torch.sum(loss)


def pairwiseLoss(z_i, z_j, pair_label):
    """ Learning objective: dot product of embeddings ~ 1_(i and j bind same target) """
    prod = torch.sigmoid(torch.bmm(z_i.unsqueeze(1), z_j.unsqueeze(2)).squeeze())
    CE = F.binary_cross_entropy(prod, pair_label)
    # print(prod, pair_label)
    return CE

def CbASLoss(out, indices, mu, logvar, w, beta=0.2):
    """ 
    CbAS loss function for VAE : weighted sum of - w_i * ELBO(x_i) 
    w : a tensor of shape (N,)
    """
    
    CE = F.cross_entropy(out, indices, reduction="none")
    CE = torch.mean(CE, dim = 1) # shape (N,)
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 2).squeeze() # to shape (N,)
    
    #print('CE :', CE)
    #print('KL :', KL)
    
    l = torch.sum(w*(CE + beta*KL))

    return l # elementwise product // 0.5 is the same KL weight as used for VAE training, otherwise KL vanishing and poor reconstruction
