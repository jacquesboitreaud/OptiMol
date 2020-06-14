# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:44:23 2019

@author: jacqu

Graph to sequence molecular VAE
RGCN encoder, GRU decoder to SELFIES 

RGCN layer at 
https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py


"""

import os
import sys
import numpy as np
from queue import PriorityQueue
import json
from rdkit import Chem
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import mean_nodes
from dgl import function as fn
from dgl.nn.pytorch.glob import SumPooling
from dgl.nn.pytorch.conv import GATConv, RelGraphConv

from utils import *
from dgl_utils import send_graph_to_device


class MultiGRU(nn.Module):
    """ 
    three layer GRU cell including an embedding layer
    and an output linear layer back to the size of the vocabulary
    """

    def __init__(self, voc_size, latent_size, h_size, dropout_rate):
        super(MultiGRU, self).__init__()
        
        p=dropout_rate # Decoder GRU dropout rate (after each layer)
        
        self.h_size = h_size
        self.dense_init = nn.Linear(latent_size, 3 * self.h_size)  # to initialise hidden state

        self.gru_1 = nn.GRUCell(voc_size, self.h_size)
        self.d1 = nn.Dropout(p)
        self.gru_2 = nn.GRUCell(self.h_size, self.h_size)
        self.d2 = nn.Dropout(p)
        self.gru_3 = nn.GRUCell(self.h_size, self.h_size)
        self.d3 = nn.Dropout(p)
        self.linear = nn.Linear(self.h_size, voc_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, h):
        """ Forward pass to 3-layer GRU. Output =  output, hidden state of layer 3 """
        x = x.view(x.shape[0], -1)  # batch_size *
        h_out = torch.zeros(h.size()).to(self.device)
        x = h_out[0] = self.gru_1(x, h[0])
        x=self.d1(x)
        x = h_out[1] = self.gru_2(x, h[1])
        x=self.d2(x)
        x = h_out[2] = self.gru_3(x, h[2])
        x=self.d3(x)
        x = self.linear(x)
        return x, h_out

    def init_h(self, z):
        """ Initializes hidden state for 3 layers GRU with latent vector z """
        batch_size, latent_shape = z.size()
        hidden = self.dense_init(z).view(3, batch_size, self.h_size)
        return hidden


class RGCN(nn.Module):
    """ RGCN encoder with num_hidden_layers + 2 RGCN layers, and sum pooling. """

    def __init__(self, features_dim, h_dim, num_rels, num_layers, num_bases=-1):
        super(RGCN, self).__init__()

        self.features_dim, self.h_dim = features_dim, h_dim
        self.num_layers = num_layers

        self.num_rels = num_rels
        self.num_bases = num_bases
        # create rgcn layers
        self.build_model()
        self.pool = SumPooling()
        
        self.p = 0.2

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = RelGraphConv(self.features_dim, self.h_dim, self.num_rels, activation=nn.ReLU(), dropout = self.p )
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_layers - 2):
            h2h = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU(), dropout = self.p)
            self.layers.append(h2h)
        # hidden to output
        h2o = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, activation=nn.ReLU(), dropout = self.p)
        self.layers.append(h2o)

    def forward(self, g):
        sequence = []
        for i, layer in enumerate(self.layers):
            # Node update 
            g.ndata['h'] = layer(g, g.ndata['h'], g.edata['one_hot'])
            # Jumping knowledge connexion
            sequence.append(g.ndata['h'])
        # Concatenation :
        g.ndata['h'] = torch.cat(sequence, dim=1)  # Num_nodes * (h_dim*num_layers)
        out = self.pool(g, g.ndata['h'].view(len(g.nodes), -1, self.h_dim * self.num_layers))
        return out


class Model(nn.Module):
    def __init__(self,
                 features_dim,
                 num_rels,
                 l_size,
                 voc_size,
                 max_len,
                 N_properties,
                 N_targets,
                 index_to_char,
                 **kwargs):
        super(Model, self).__init__()

        # params:

        # Encoding
        self.features_dim = features_dim
        self.gcn_hdim = kwargs['gcn_hdim']
        self.gcn_layers = kwargs['gcn_layers']
        self.num_rels = num_rels

        # Bottleneck
        self.l_size = l_size

        # Decoding
        self.gru_hdim = kwargs['gru_hdim']
        self.dropout = kwargs['gru_dropout']
        self.voc_size = voc_size
        self.max_len = max_len
        self.index_to_char = index_to_char

        self.N_properties = N_properties
        self.N_targets = N_targets

        # layers:
        self.encoder = RGCN(self.features_dim, self.gcn_hdim, self.num_rels, self.gcn_layers, num_bases=-1)

        self.encoder_mean = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)
        self.encoder_logv = nn.Linear(self.gcn_hdim * self.gcn_layers, self.l_size)

        self.rnn_in = nn.Linear(self.l_size, self.voc_size)
        self.decoder = MultiGRU(voc_size=self.voc_size, latent_size=self.l_size, h_size=self.gru_hdim, dropout_rate = self.dropout)

        # MOLECULAR PROPERTY REGRESSOR
        self.MLP = nn.Sequential(
            nn.Linear(self.l_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.N_properties))


    @property
    def device(self):
        return next(self.parameters()).device

    def load(self, trained_path, aff_net=False):
        # Loads trained model weights, with or without the affinity predictor
        if aff_net:
            self.load_state_dict(torch.load(trained_path))
        else:
            self.load_no_multitask(trained_path)
        # print(f'Loaded weights from {trained_path}')

    # ======================== Model pass functions ==========================

    def forward(self, g, smiles, tf):
        # print('edge data size ', g.edata['one_hot'].size())
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z = self.sample(mu, logv, mean_only=False).squeeze()  # stochastic sampling
        out = self.decode(z, smiles, teacher_forced=tf)  # teacher forced decoding
        properties = self.MLP(z)

        return mu, logv, z, out, properties

    def sample(self, mean, logv, mean_only):
        """
         Samples a vector according to the latent vector mean and variance
        :param mean:
        :param logv:
        :param mean_only:
        :return:
        """
        if not mean_only:
            sigma = torch.exp(.5 * logv)
            return mean + torch.randn_like(mean) * sigma
        else:
            return mean

    def encode(self, g, mean_only):
        """ Encodes to latent space, with or without stochastic sampling """
        e_out = self.encoder(g)
        mu, logv = self.encoder_mean(e_out), self.encoder_logv(e_out)
        z = self.sample(mu, logv, mean_only).squeeze()  # train to true for stochastic sampling
        return z

    def props(self, z):
        # Returns predicted properties
        return self.MLP(z)

    def decode(self, z, x_true=None, teacher_forced=0.0):
        """
            Unrolls decoder RNN to generate a batch of sequences, using teacher forcing
            Args:
                z: (batch_size * latent_shape) : a sampled vector in latent space
                x_true: (batch_size * sequence_length ) a batch of indices of sequences 
            Outputs:
                gen_seq : (batch_size * voc_size* seq_length) a batch of generated sequences (probas)
                
        """
        batch_size = z.shape[0]
        # ls= z.shape[1]
        # print('batch size is', batch_size, 'latent size is ', ls)
        seq_length = self.max_len
        # Create first input to RNN : start token is full of zeros
        start_token = self.rnn_in(z).view(batch_size, 1, self.voc_size)
        rnn_in = start_token.to(self.device)
        # Init hidden with z sampled in latent space 
        h = self.decoder.init_h(z)

        gen_seq = torch.zeros(batch_size, self.voc_size, seq_length).to(self.device)
        
        #tback = time.perf_counter()
        
        for step in range(seq_length):
            out, h = self.decoder(rnn_in, h)
            gen_seq[:, :, step] = out

            if teacher_forced > 0.0 and np.random.rand() < teacher_forced:  # proba of teacher forcing
                indices = x_true[:, step]
            else:
                v, indices = torch.max(gen_seq[:, :, step], dim=1)  # get char indices with max probability
            # Input to next step: either autoregressive or Teacher forced
            rnn_in = F.one_hot(indices, self.voc_size).float()
            
        #if torch.cuda.is_available():
        #    torch.cuda.synchronize()
        #print(f'time in rnn: {time.perf_counter() - tback}')

        return gen_seq  # probas

    def probas_to_smiles(self, gen_seq):
        # Takes tensor of shape (N, voc_size, seq_len), returns list of corresponding smiles
        N, voc_size, seq_len = gen_seq.shape
        v, indices = torch.max(gen_seq, dim=1)
        indices = indices.cpu().numpy()
        smiles = []
        for i in range(N):
            smiles.append(''.join([self.index_to_char[str(idx)] for idx in indices[i]]).rstrip())
        return smiles

    def fix_index_to_char(self):
        self.index_to_char = {str(k): v for k, v in self.index_to_char.items()}
        print('fixed idx to char')

    def indices_to_smiles(self, indices):
        # Takes indices tensor of shape (N, seq_len), returns list of corresponding smiles
        N, seq_len = indices.shape
        try:
            indices = indices.cpu().numpy()
        except:
            pass
        smiles = []
        for i in range(N):
            smiles.append(''.join([self.index_to_char[str(idx)] for idx in indices[i]]).rstrip())
        return smiles

    def beam_out_to_smiles(self, indices):
        """ Takes array of possibilities : (N, k_beam, sequences_length)  returned by decode_beam"""
        N, k_beam, length = indices.shape
        smiles = []
        for i in range(N):
            k, m = 0, None
            while (k < 2 and m == None):
                smi = ''.join([self.index_to_char[str(idx)] for idx in indices[i, k]])
                smi = smi.rstrip()
                m = Chem.MolFromSmiles(smi)
                k += 1
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
        if cutoff_mols != None:
            N = cutoff_mols
            print(f'Decoding will stop after {N} mols')
        sequences = []
        for n in range(N):
            print("decoding molecule n° ", n)
            # Initialize rnn states and input
            z_1mol = z[n].view(1, self.l_size)  # Reshape as a batch of size 1
            start_token = self.rnn_in(z_1mol).view(1, self.voc_size, 1).to(self.device)
            rnn_in = start_token
            h = self.decoder.init_h(z_1mol)
            topk = [BeamSearchNode(h, rnn_in, 0, [])]

            for step in range(self.max_len):
                next_nodes = PriorityQueue()
                for candidate in topk:  # for each candidate sequence (among k)
                    score = candidate.score
                    seq = candidate.sequence
                    # pass into decoder
                    out, new_h = self.decoder(candidate.rnn_in, candidate.h)
                    probas = F.softmax(out, dim=1)  # Shape N, voc_size
                    for c in range(self.voc_size):
                        new_seq = seq + [c]
                        rnn_in = torch.zeros((1, 36))
                        rnn_in[0, c] = 1
                        s = score - probas[0, c]
                        next_nodes.put((s.item(), BeamSearchNode(new_h, rnn_in.to(self.device), s.item(), new_seq)))
                topk = []
                for k_ in range(k):
                    # get top k for next timestep !
                    score, node = next_nodes.get()
                    topk.append(node)
                    # print("top sequence for next step :", node.sequence)

            sequences.append([n.sequence for n in topk])  # list of lists
        return np.array(sequences)

    # ========================== Sampling functions ======================================

    def sample_around_mol(self, g, dist, beam_search=False, attempts=1, props=False, aff=False):
        """ Samples around embedding of molecular graph g, within a l2 distance of d """
        e_out = self.encoder(g)
        mu, var = self.encoder_mean(e_out), self.encoder_logv(e_out)
        sigma = torch.exp(.5 * var)

        tensors_list = []
        for i in range(attempts):
            noise = torch.randn_like(mu) * sigma
            noise = (dist / torch.norm(noise, p=2, dim=1)) * noise  # rescale noise norm to be equal to dist
            noise = noise.to(self.device)
            sp = mu + noise
            tensors_list.append(sp)

        if attempts > 1:
            samples = torch.stack(tensors_list, dim=0)
            samples = torch.squeeze(samples)
        else:
            samples = sp

        if beam_search:
            dec = self.decode_beam(samples)
        else:
            dec = self.decode(samples)

        # props ad affinity if requested 
        p, a = 0, 0
        if props:
            p = self.props(samples)
        if aff:
            a = self.aff(samples)

        return dec, p, a

    def sample_around_z(self, z, dist, beam_search=False, attempts=1, props=False, aff=False):
        """ Samples around embedding of molecular graph g, within a l2 distance of d """

        sigma = torch.exp(.5 * torch.randn_like(z)).to(self.device)
        z = z.to(self.device)
        tensors_list = []
        for i in range(attempts):
            noise = torch.randn_like(z) * sigma
            noise = (dist / torch.norm(noise, p=2, dim=1)) * noise  # rescale noise norm to be equal to dist
            noise = noise.to(self.device)
            sp = z + noise
            tensors_list.append(sp)

        if attempts > 1:
            samples = torch.stack(tensors_list, dim=0)
            samples = torch.squeeze(samples)
        else:
            samples = sp
        """
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
        """
        return samples

    def sample_z_prior(self, n_mols):
        """Sampling z ~ p(z) = N(0, I)
        :param n_batch: number of batches
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        latent = torch.normal(mean=0., std=1., size=(n_mols, self.l_size))
        # latent_points = []
        # for i in range(n_mols):
        #     latent_points.append(torch.normal(torch.zeros(self.l_size), torch.ones(self.l_size)).view(1, self.l_size))
        #
        # latent = torch.cat(latent_points, dim=0)

        return latent.to(self.device)

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
        batch_size = loader.batch_size

        # Latent embeddings
        z_all = []

        with torch.no_grad():
            for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
                # batch_size = graph.batch_size
                graph = send_graph_to_device(graph, self.device)

                z = self.encode(graph, mean_only=True)  # z_shape = N * l_size
                z = z.cpu()
                z_all.append(z)

        z_all = torch.cat(z_all, dim=0).numpy()
        return z_all

    def load_no_multitask(self, state_dict):
        # Workaround to be able to load a model with not same size of affinity predictor... 
        pretrained_dict = torch.load(state_dict)
        model_dict = self.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)


class BeamSearchNode():
    def __init__(self, h, rnn_in, score, sequence):
        self.h = h
        self.rnn_in = rnn_in
        self.score = score
        self.sequence = sequence
        self.max_len = 60

    def __lt__(self, other):  # For x < y
        # Pour casser les cas d'égalité du score au hasard, on s'en fout un peu.
        # Eventuellement affiner en regardant les caractères de la séquence (pénaliser les cycles ?)
        return True


def model_from_json(name='inference_default', load_weights=True, default_dir=True, weights_path=None):
    """
    Load a model from the name of the experiment
    :param name:
    :param load_weights:
    :return:
    """
    dumper = ModelDumper()
    if default_dir:
        path_to_dir = os.path.join(script_dir, 'results/saved_models', name)
        params = dumper.load(os.path.join(path_to_dir, 'params.json'))
    else:
        params = dumper.load(name)

    model = Model(**params)
    if load_weights:
        try:
            if default_dir:
                model.load(os.path.join(path_to_dir, "weights.pth"))
            else:
                model.load(weights_path)

        except:
            print('Weights could not be loaded by the util functions')
    return model


def model_from_dir(dir, load_weights=True):
    dumper = ModelDumper()
    params = dumper.load(os.path.join(dir, 'params.json'))
    model = Model(**params)
    if load_weights:
        try:
            model.load(os.path.join(dir, "weights.pth"))
        except:
            print('Weights could not be loaded by the util functions')
    return model
