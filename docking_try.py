# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Loads pretrained multitask VAE and iterates sampling and docking to improve the affinity predictor in latent space. 
(TODO)

Args : 
    
    --model : path to pretrained VAE weights. 
    
Questions : 
    end-to-end or freeze the VAE and just learn the predictor network ? 


"""

import argparse
import sys, os
import torch
import numpy as np

import pickle
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import *

from selfies import decoder

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, 'dataloaders'))
    sys.path.append(os.path.join(script_dir, 'data_processing'))
    
    sys.path.append(os.path.join(script_dir, '..', 'vina_docking')) # make sur to clone 'vina_docking' repo. 
    
    from model import Model, Loss, multiLoss
    from dataloaders.molDataset import molDataset, Loader
    from dock_AL_batch import dock_batch # docking function 

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='saved_model_w/g2s_iter_40000.pth') # Path to pretrained VAE 

    parser.add_argument('--decode', type=str, default='selfies') # 'smiles' or 'selfies'

    parser.add_argument('--lr', type=float, default=1e-3) # Initial learning rate

    parser.add_argument('--processes', type=int, default=8) # num workers 

     # =======

    args=parser.parse_args()
    
    disable_rdkit_logging() # function from utils to disable rdkit logs

    #Loading model 

    params = pickle.load(open('saved_model_w/model_params.pickle','rb'))
    model = Model(**params)
    model.load_state_dict(torch.load(args.model))
    print('> Loaded pretrained VAE')
    device = model.device
    model.to(device)

    print(model)
    
    top_samples = torch.zeros(100, 64) # n_mols * latent size 
    top_preds = torch.zeros(100)
    

    for i in range(100):
        
        samp = model.sample_z_prior(n_mols = 10)
        #pred_a = model.aff_net(samp)
        
        # ============================================================
        # Active learning iteration
        # ============================================================
        
        # Acquisition function 
        # Exploitation: take the highest predicted aff in batch 
        #_, argmax = torch.max(pred_a, dim=0)
        
        # Exploration: random 
        argmax = np.random.randint(0, high = 10)
        top_samples[i,:] =  samp[argmax, :]
            
    top_samples = top_samples.to(device)
    smiles = model.decode(top_samples)
    smiles = model.probas_to_smiles(smiles)
    smiles = [decoder(s) for s in smiles]
    
    # Request scores
    scores = dock_batch(smiles)
    
    for i in range(100):
        print('smiles: ', smiles[i])
        print('true score: ', scores[i] )
    
    # Save scores to known values 
    
    # Compute affinity prediction loss on these values 
    #regression_loss = F.mse_loss(top_preds, torch.tensor(scores), reduction="sum")
    #print('MSE loss: ', regression_loss.item())



