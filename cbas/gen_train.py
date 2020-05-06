# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:46:45 2020

@author: jacqu

Function that trains a model on samples x_i, weights w_i and returns the decoder parameters phi. 
"""

import os , sys 

import torch 
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from model import model_from_json
from loss_func import CbASLoss
from dataloaders.simple_loader import SimpleLoader
from utils import *

def GenTrain(x ,w , model, savepath, epochs = 10):
    """
    Input : x a list of M inputs (smiles)
            w an array of shape (M,) with weights assigned to each of the samples
            init_model_dir : dir with initial model weights 
            epochs : nbr of training epochs 
            model_dir : directory to save update weights at the end of training 
            
    Output : None 
    """
    
    device = model.device
    
    teacher_forcing = 0 
    
    # Optimizer 
    lr0 = 1e-3
    anneal_rate = 0.9
    anneal_iter = 40000
    optimizer = optim.Adam(model.parameters(), lr=lr0)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    
    # loader 
    train_loader = SimpleLoader(x).get_data() # loader for given smiles 

    # Training loop
    total_steps=0 
    for epoch in range(epochs):
        print(f'Starting epoch {epoch}')
        model.train()
        
        for batch_idx, (graph, smiles, _, _) in enumerate(train_loader):
            total_steps += 1  # count training steps

            smiles = smiles.to(device)
            graph = send_graph_to_device(graph, device)

            # Forward pass
            mu, logv, _, out_smi, out_p, out_a = model(graph, smiles, tf=teacher_forcing) # no tf 

            # Compute CbAS loss with samples weights 
            loss = CbASLoss(out_smi, smiles, mu, logv, w)

            optimizer.zero_grad()
            loss.backward()
            del loss
            clip.clip_grad_norm_(model.parameters(), 50)
            optimizer.step()

            # Annealing KL and LR
            if total_steps % anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])
                
    
    # Update weights at 'save_model_weights' : 
    print(f'Finished training at step {total_steps}. Saving model weights')
    model.cpu()
    torch.save(model.state_dict(), os.path.join(search_model, "weights.pth"))
    
    return

        
        
    
    
    
    


