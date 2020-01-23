# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

RGCN encoder, GRU decoder to can smiles 
https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py


"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import itertools
from queue import PriorityQueue

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
    def __init__(self, voc_size, latent_size, h_size):
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
        self.max_len = 151
        
        self.N_properties=N_properties
        self.N_targets = N_targets
        
        self.device = device
        
        # layers:
        self.encoder=RGCN(self.features_dim, self.gcn_hdim, self.gcn_outdim, self.num_rels, 
                          num_bases=-1, num_hidden_layers=2).to(self.device)
        
        self.encoder_mean = nn.Linear(self.gcn_outdim , self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_outdim , self.l_size)
        
        self.rnn_in= nn.Linear(self.l_size,self.voc_size)
        self.decoder = MultiGRU(voc_size=self.voc_size, latent_size= self.l_size, h_size=400)
        
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
        
    def sample(self, mean, logv, mean_only):
        """ Samples a vector according to the latent vector mean and variance """
        if not mean_only:
            sigma = torch.exp(.5 * logv)
            return mean + torch.randn_like(mean) * sigma
        else:
            return mean
        
    def encode(self, g, mean_only):
        """ Encodes to latent space, with or without stochastic sampling """
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z= self.sample(mu, logv, mean_only).squeeze() # train to true for stochastic sampling 
        return z
    
    def props(self,z):
        return self.MLP(z)
    
    def affs(self,z):
        return self.aff_net(z)
        
    def decode(self, z, x_true=None,teacher_forced=False):
        """
            Unrolls decoder RNN to generate a batch of sequences, using teacher forcing
            Args:
                z: (batch_size * latent_shape) : a sampled vector in latent space
                x_true: (batch_size * sequence_length ) a batch of indices of sequences 
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
            
            if(teacher_forced):
                indices = x_true[:,step]
            else:
                v, indices = torch.max(gen_seq[:,:,step],dim=1) # get char indices with max probability
            # Input to next step: either autoregressive or Teacher forced
            rnn_in =F.one_hot(indices,self.voc_size).float()
                
        return gen_seq
    
    def decode_beam(self, z, k=3, cutoff_mols=None):
        """
        Input:
            z = torch.tensor type, (N_mols*l_size)  
            k : beam param
        Decodes a batch, molecule by molecule, using beam search of width k 
        Output: 
            a list of lists of k best sequences for each molecule.
        """
        N = z.shape[0]
        if(cutoff_mols!=None):
            N=cutoff_mols
            print(f'Decoding will stop after {N} mols')
        sequences = []
        for n in range(N):
            print("decoding molecule n° ",n)
            # Initialize rnn states and input
            z_1mol=z[n].view(1,self.l_size) # Reshape as a batch of size 1
            start_token = self.rnn_in(z_1mol).view(1,self.voc_size,1).to(self.device)
            rnn_in = start_token
            h = self.decoder.init_h(z_1mol)
            topk = [BeamSearchNode(h,rnn_in, 0, [] )]
            
            for step in range(self.max_len):
                next_nodes=PriorityQueue()
                for candidate in topk: # for each candidate sequence (among k)
                    score = candidate.score
                    seq=candidate.sequence
                    # pass into decoder
                    out, new_h = self.decoder(candidate.rnn_in, candidate.h) 
                    probas = F.softmax(out, dim=1) # Shape N, voc_size
                    for c in range(self.voc_size):
                        new_seq=seq+[c]
                        rnn_in=torch.zeros((1,36))
                        rnn_in[0,c]=1
                        s= score-probas[0,c]
                        next_nodes.put(( s.item(), BeamSearchNode(new_h, rnn_in.to(self.device),s.item(), new_seq)) )
                topk=[]
                for k_ in range(k):
                    # get top k for next timestep !
                    score, node=next_nodes.get()
                    topk.append(node)
                    #print("top sequence for next step :", node.sequence)
                    
            sequences.append([n.sequence for n in topk]) # list of lists 
        return np.array(sequences)
        
    def forward(self, g, smiles):
        #print('edge data size ', g.edata['one_hot'].size())
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z= self.sample(mu, logv, mean_only=False).squeeze() # stochastic sampling 
        out = self.decode(z, smiles, teacher_forced=True) # teacher forced decoding 
        properties = self.MLP(z)
        affinities = self.aff_net(z)
        
        return mu, logv,z, out, properties, affinities
    

    
def Loss(out, indices, mu, logvar, y_p, p_pred,
         y_a, a_pred, train_on_aff):
    """ 
    Loss function for multitask VAE. uses Crossentropy for reconstruction
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    mse= F.mse_loss(p_pred, y_p, reduction="sum")
    
    #affinities: 
    if(train_on_aff):
        aff_loss = F.mse_loss(a_pred,y_a,reduction='sum') # weighted loss 
    else: 
        aff_loss = torch.tensor(0) 
    
    return CE, KLD, mse, aff_loss # returns 4 values

def tripletLoss(z_i, z_j, z_l, margin=2):
    
    print(z_i.shape)
    dij = torch.norm(z_i-z_j, p=2, dim=1) # z vectors are (N*l_size), compute norm along latent size, for each batch item.
    dil = torch.norm(z_i-z_l, p=2, dim=1)
    loss = torch.max(torch.cuda.FloatTensor(z_i.shape[0]).fill_(0), dij -dil + margin)
    print(loss.shape)
    # Embeddings distance loss 
    return torch.sum(loss)

def RecLoss(out, indices):
    """ 
    Crossentropy for SMILES reconstruction 
    out : (N, n_chars, l), where l = sequence length
    indices : (N, l)
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    return CE