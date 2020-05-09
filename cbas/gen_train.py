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
from torch.utils.data import Dataset, DataLoader

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from model import model_from_json
from loss_func import CbASLoss
from dataloaders.simple_loader import SimpleDataset, collate_block
from utils import *
from dgl_utils import *

class GenTrain():
    """ 
    Wrapper for search model iterative training in CbAS
    """

    def __init__(self, model_init_path, savepath, epochs, device, lr, clip_grad, processes = 8):
        super(GenTrain, self).__init__()
        
        self.model = model_from_json(model_init_path)
        self.savepath =savepath
        soft_mkdir(self.savepath) # create dir to save the search model 
        self.device = device
        self.model.to(self.device)
        self.n_epochs = epochs
        
        self.processes = processes
        
        self.teacher_forcing = 0 
        # Optimizer 
        self.lr0 = lr
        self.anneal_rate = 0.9
        self.anneal_iter = 40000
        self.clip_grads = clip_grad
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr0)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, self.anneal_rate)
        
        # loader 
        self.dataset = SimpleDataset(maps_path = '../map_files', vocab = 'selfies')
        
    def step(self, input_type, x , w):
        """ 
        Trains the model for n_epochs on samples x, weighted by w 
        input type : 'selfies' or 'smiles', for dataloader (validity checks and format conversions are different)
        """
        
        if input_type =='smiles':
            self.dataset.pass_smiles_list( x,w)
        elif input_type =='selfies':
            self.dataset.pass_selfies_list( x,w)
        
        train_loader = DataLoader(dataset=self.dataset, shuffle=True, batch_size=64,
                                      num_workers=self.processes, collate_fn=collate_block, drop_last=True)
        # Training loop
        total_steps=0 
        for epoch in range(self.n_epochs):
            print(f'Starting epoch {epoch}')
            self.model.train()
            
            for batch_idx, (graph, smiles, w_i) in enumerate(train_loader):
                total_steps += 1  # count training steps
    
                smiles = smiles.to(self.device)
                graph = send_graph_to_device(graph, self.device)
                w_i=w_i.to(self.device)
    
                # Forward pass
                mu, logv, _, out_smi, out_p, out_a = self.model(graph, smiles, tf=self.teacher_forcing) # no tf 
    
                # Compute CbAS loss with samples weights 
                loss = CbASLoss(out_smi, smiles, mu, logv, w_i)
                if(batch_idx==0):
                    print(f'CbAS Loss at batch 0 : {loss.item()}')
    
                self.optimizer.zero_grad()
                loss.backward()
                del loss
                clip.clip_grad_norm_(self.model.parameters(), self.clip_grads)
                self.optimizer.step()
    
                # Annealing KL and LR
                if total_steps % self.anneal_iter == 0:
                    self.scheduler.step()
                    print("learning rate: %.6f" % self.scheduler.get_lr()[0])
                    
        
        # Update weights at 'save_model_weights' : 
        print(f'Finished training after {total_steps} optimizer steps. Saving search model weights')
        self.model.cpu()
        torch.save(self.model.state_dict(), os.path.join(self.savepath, "weights.pth"))
        self.model.to(self.device)

        
        
    
    
    
    


