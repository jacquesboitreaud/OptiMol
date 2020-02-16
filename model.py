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
import json

from rdkit import Chem

import dgl
from dgl import mean_nodes
from dgl import function as fn
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

from utils import *
import pickle

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
                 device,
                 binary_labels = True):
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
        self.binary_labels = binary_labels
        
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
            
        # Affinities predictor (If binary, sigmoid applied directly in fwd function)
        self.aff_net = nn.Sequential(
                nn.Linear(self.l_size,32),
                nn.ReLU(),
                nn.Linear(32,32),
                nn.ReLU(),
                nn.Linear(32,self.N_targets))
        
    def load(self, trained_path):
        # Loads trained model weights 
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        params = pickle.load(open('saved_model_w/params.pickle','rb'))
        self.load_state_dict(torch.load(trained_path))
        self.to(device)
        print(f'Loaded weights from {trained_path} to {device}')
        
        return device
        
    def set_smiles_chars(self,char_file="map_files/zinc_chars.json"):
        # Adds dict to convert indices to smiles chars 
        self.char_list = json.load(open(char_file))
        self.char_to_index= dict((c, i) for i, c in enumerate(self.char_list))
        self.index_to_char= dict((i, c) for i, c in enumerate(self.char_list))
        
    # ======================== Model pass functions ==========================
    
    def forward(self, g, smiles):
        #print('edge data size ', g.edata['one_hot'].size())
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z= self.sample(mu, logv, mean_only=False).squeeze() # stochastic sampling 
        out = self.decode(z, smiles, teacher_forced=True) # teacher forced decoding 
        properties = self.MLP(z)
        affinities = self.aff_net(z)
        if(self.binary_labels):
            affinities = torch.sigmoid(affinities)
        
        return mu, logv,z, out, properties, affinities
        
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
        a = self.aff_net(z)
        if(self.binary_labels):
            return torch.sigmoid(a)
        else:
            return a 
        
    def decode(self, z, x_true=None,teacher_forced=False):
        """
            Unrolls decoder RNN to generate a batch of sequences, using teacher forcing
            Args:
                z: (batch_size * latent_shape) : a sampled vector in latent space
                x_true: (batch_size * sequence_length ) a batch of indices of sequences 
            Outputs:
                gen_seq : (batch_size * voc_size* seq_length) a batch of generated sequences (probas)
                
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
                
        return gen_seq # probas 
    
    def probas_to_smiles(self, gen_seq):
        # Takes tensor of shape (N, voc_size, seq_len), returns list of corresponding smiles
        N, voc_size, seq_len = gen_seq.shape
        v, indices = torch.max(gen_seq, dim=1)
        indices = indices.cpu().numpy()
        smiles = []
        for i in range(N):
            smiles.append(''.join([self.index_to_char[idx] for idx in indices[i]]).rstrip())
        return smiles
    
    def indices_to_smiles(self, indices):
        # Takes indices tensor of shape (N, seq_len), returns list of corresponding smiles
        N, seq_len = indices.shape
        try:
            indices = indices.cpu().numpy()
        except:
            pass
        smiles = []
        for i in range(N):
            smiles.append(''.join([self.index_to_char[idx] for idx in indices[i]]).rstrip())
        return smiles
    
    def beam_out_to_smiles(self,indices):
        """ Takes array of possibilities : (N, k_beam, sequences_length)  returned by decode_beam"""
        N, k_beam, length = indices.shape
        smiles = []
        for i in range(N):
            k,m = 0, None 
            while(k<2 and m==None):
                smi = ''.join([self.index_to_char[idx] for idx in indices[i,k]])
                smi = smi.rstrip()
                m=Chem.MolFromSmiles(smi)
                k+=1
            smiles.append(smi)
            print(smi)
        return smiles
    
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
    
    # ========================== Sampling functions ======================================
    
    def sample_around_mol(self, g, dist, beam_search=False, attempts = 1, props=False, aff=False):
        """ Samples around embedding of molecular graph g, within a l2 distance of d """
        e_out = self.encoder(g)
        mu, var = self.encoder_mean(e_out), self.encoder_logv(e_out)
        sigma = torch.exp(.5 * var)
        
        tensors_list = []
        for i in range(attempts):
            noise = torch.randn_like(mu) * sigma
            noise = (dist/torch.norm(noise,p=2,dim=1))*noise
            noise = noise.to(self.device)
            sp=mu + noise 
            tensors_list.append(sp)
        
        if(attempts>1):
            samples=torch.stack(tensors_list, dim=0)
            samples = torch.squeeze(samples)
        else:
            samples = sp
            
        if(beam_search):
            dec = self.decode_beam(samples)
        else:
            dec = self.decode(samples)
            
        # props ad affinity if requested 
        p,a = 0,0
        if(props):
            p = self.props(samples)
        if(aff):
            a = self.aff(samples)
        
        return dec, p, a
    
    def sample_z_prior(self, n_mols):
        """Sampling z ~ p(z) = N(0, I)
        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """
        return torch.randn(n_mols, self.l_size, device=self.device)
    
    # ========================= Packaged functions to use trained model ========================
    
    def embed(self, loader, df):
    # Gets latent embeddings of molecules in df. 
    # Inputs : 
    # 0. loader object to convert smiles into batches of inputs 
    # 1. dataframe with 'can' column containing smiles to embed 
    # Outputs :
    # 0. np array of embeddings, (N_molecules , latent_size)
    
        loader.dataset.pass_dataset(df)
        _, _, test_loader = loader.get_data()
        batch_size=loader.batch_size
        
        # Latent embeddings
        z_all = torch.zeros(loader.dataset.n,self.l_size)
        
        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
                
                graph=send_graph_to_device(graph,self.device)
            
                z = self.encode(graph, mean_only=True) #z_shape = N * l_size
                z=z.cpu()
                z_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=z
                
        z_all = z_all.numpy()
        return z_all
        
# ======================= Loss functions ====================================
    
def Loss(out, indices, mu, logvar, y_p, p_pred,
         y_a, a_pred, train_on_aff, binary_aff=False):
    """ 
    Loss function for VAE + multitask.
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Weighted mse loss (100*qed + 10 logp + 0.1 molweight)
    mse= 100*F.mse_loss(p_pred[:,0], y_p[:,0], reduction="sum") +\
    10*F.mse_loss(p_pred[:,1], y_p[:,1], reduction="sum") + 0.1* F.mse_loss(p_pred[:,2], y_p[:,2], reduction="sum")
    
    #affinities: 
    if(train_on_aff and binary_aff):
        aff_loss = F.cross_entropy(a_pred, y_a, reduction="sum") # binary binding labels
    elif(train_on_aff):
        aff_loss = F.mse_loss(a_pred,y_a,reduction='sum') # regression IC50 values 
    else: 
        aff_loss = torch.tensor(0) # No affinity prediction 
    
    return CE, KLD, mse, aff_loss # returns 4 values

def tripletLoss(z_i, z_j, z_l, margin=2):
    """ For disentangling by separating known actives in latent space """
    dij = torch.norm(z_i-z_j, p=2, dim=1) # z vectors are (N*l_size), compute norm along latent size, for each batch item.
    dil = torch.norm(z_i-z_l, p=2, dim=1)
    loss = torch.max(torch.cuda.FloatTensor(z_i.shape[0]).fill_(0), dij -dil + margin)
    # Embeddings distance loss 
    return torch.sum(loss)

def pairwiseLoss(z_i,z_j,pair_label):
    """ Learning objective: dot product of embeddings ~ 1_(i and j bind same target) """
    prod = torch.sigmoid(torch.bmm(z_i.unsqueeze(1),z_j.unsqueeze(2)).squeeze())
    CE = F.binary_cross_entropy(prod, pair_label)
    #print(prod, pair_label)
    return CE

def RecLoss(out, indices):
    """ 
    Only crossentropy for SMILES reconstruction 
    out : (N, n_chars, l), where l = sequence length
    indices : (N, l)
    """
    CE = F.cross_entropy(out, indices, reduction="sum")
    return CE