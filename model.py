# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

RGCN encoder, GRU decoder to can smiles 
https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py


"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import itertools

import dgl
from dgl import mean_nodes
from dgl import function as fn
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

from utils import *

class MultiGRU(nn.Module):
    """ 
    three layer GRU cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """
    def __init__(self, voc_size, latent_size, h_size=100):
        super(MultiGRU, self).__init__()
        
        self.h_size=h_size
        self.dense_init=nn.Linear(latent_size,3*self.h_size) # to initialise hidden state
        
        self.gru_1 = nn.GRUCell(voc_size, self.h_size)
        self.gru_2 = nn.GRUCell(self.h_size, self.h_size)
        self.gru_3 = nn.GRUCell(self.h_size, self.h_size)
        self.linear = nn.Linear(self.h_size, voc_size)

    def forward(self, x, h):
        """ Forward pass to 3-layer GRU. Output =  output, hidden state of layer 3 """
        x = x.view(x.shape[0],-1) # batch_size * 128
        h_out = Variable(torch.zeros(h.size()))
        x= h_out[0] = self.gru_1(x, h[0])
        x= h_out[1] = self.gru_2(x, h[1])
        x= h_out[2] = self.gru_3(x, h[2])
        x = self.linear(x)
        return x, h_out

    def init_h(self, z):
        """ Initializes hidden state for 3 layers GRU with latent vector z """
        batch_size, latent_shape = z.size()
        hidden=self.dense_init(z).view(3,batch_size, self.h_size)
        return hidden


class RGCN(nn.Module):
    """ RGCN encoder with num_hidden_layers + 2 RGCN layers, and sum pooling. """
    def __init__(self, features_dim, h_dim, out_dim , num_rels, num_bases=-1, num_hidden_layers=2):
        super(RGCN, self).__init__()
        
        self.features_dim, self.h_dim, self.out_dim = features_dim, h_dim, out_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()
        self.pool = SumPooling()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU())
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.out_dim, self.num_rels, activation=nn.ReLU())
        self.layers.append(h2o)
        
    def forward(self, g):
        #print('edge data size ', g.edata['one_hot'].size())
        for layer in self.layers:
             g.ndata['h']=layer(g,g.ndata['h'],g.edata['one_hot'])
        out=self.pool(g,g.ndata['h'].view(len(g.nodes),-1,self.out_dim))
        return out
    
class Model(nn.Module):
    def __init__(self, features_dim, gcn_hdim, gcn_outdim , num_rels,
                 l_size, voc_size,
                 N_properties, N_targets,
                 device):
        super(Model, self).__init__()
        
        # params:
        self.features_dim = features_dim
        self.gcn_hdim = gcn_hdim
        self.gcn_outdim = gcn_outdim
        self.num_rels = num_rels
        self.l_size = l_size
        self.voc_size = voc_size 
        self.max_len = 100
        
        self.N_properties=N_properties
        self.N_targets = N_targets
        
        self.device = device
        
        # layers:
        self.encoder=RGCN(self.features_dim, self.gcn_hdim, self.gcn_outdim, self.num_rels, 
                          num_bases=-1, num_hidden_layers=2).to(self.device)
        
        self.encoder_mean = nn.Linear(self.gcn_outdim , self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_outdim , self.l_size)
        
        self.rnn_in= nn.Linear(self.l_size,self.voc_size)
        self.decoder = MultiGRU(self.voc_size, self.l_size, 100)
        
        # MOLECULAR PROPERTY REGRESSOR
        self.MLP=nn.Sequential(
                nn.Linear(self.l_size,32),
                nn.ReLU(),
                nn.Linear(32,16),
                nn.ReLU(),
                nn.Linear(16,self.N_properties))
            
        # BINDING SCORES PREDICTOR -> pIC50 ( 0: non binder, 9: nanomolar activity)
        self.aff_net = nn.Sequential(
            nn.Linear(self.l_size,32),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Linear(32,self.N_targets))
        
    def sample(self, mean, logv, train):
        """ Samples a vector according to the latent vector mean and variance """
        if train:
            sigma = torch.exp(.5 * logv)
            return mean + torch.randn_like(mean) * sigma
        else:
            return mean
        
    def decode(self, z, x_true=None):
        """
            Unrolls decoder RNN to generate a batch of sequences, using teacher forcing
            Args:
                z: (batch_size * latent_shape) : a sampled vector in latent space
                x_true: (batch_size * sequence_length * voc_size) a batch of sequences
            Outputs:
                gen_seq : (batch_size * voc_size* seq_length) a batch of generated sequences
                
        """
        batch_size=z.shape[0]
        seq_length=self.max_len
        # Create first input to RNN : start token is full of zeros
        start_token = self.rnn_in(z).view(batch_size,1,self.voc_size)
        rnn_in = start_token.to(self.device)
        # Init hidden with z sampled in latent space 
        h = self.decoder.init_h(z)
        
        gen_seq = Variable(torch.zeros(batch_size, self.voc_size,seq_length))
        
        for step in range(seq_length):
            out, h = self.decoder(rnn_in, h) 
            gen_seq[:,:,step]=out
            
            indices = x_true[:,step]
            rnn_in =F.one_hot(indices,self.voc_size).float()
                
        return gen_seq
        
    def forward(self, g, smiles):
        #print('edge data size ', g.edata['one_hot'].size())
        
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z= self.sample(mu, logv, train=False).squeeze() # train to true for stochastic sampling 

        out = self.decode(z, smiles) # teacher forced decoding 
        
        return out
    

    
def Loss(out, indices, mu, logvar, y=None, pred_properties=None, kappa=1.0):
    """ 
    Loss function for multitask VAE. uses Crossentropy for reconstruction
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    mse= F.mse_loss(pred_properties, y, reduction="sum")
    total_loss=CE + kappa*KLD + mse
    return total_loss, CE, KLD, mse # returns 4 values

def RecLoss(out, indices):
    """ 
    Crossentropy for SMILES reconstruction 
    out : (N, n_chars, l), where l = sequence length (100)
    indices : (N, l)
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    return CE