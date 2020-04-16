# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:09:30 2019

@author: jacqu

Utils functions for working with PDB files using biopython
"""

import numpy as np
import dgl
import torch
import pandas as pd

import rdkit
from rdkit import Chem
import os 


def _make_dir(directory):
    os.makedirs(directory)
    
def disable_rdkit_logging():
    """
    Disables RDKit logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')


# ================= Pytorch utils ================================

def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape), o.size())
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    print(f"Found {len(tensors)} unique tensors. Total: {sum(tensors.values())}")
    for line in sorted(tensors.items(), key=lambda x: x[1], reverse=True):
        print('{}\t{}'.format(*line))


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if (torch.cuda.is_available()):
        return torch.autograd.Variable(tensor).cuda()
    else:
        return torch.autograd.Variable(tensor)


# ============== Smiles handling utils ===============================

def log_reconstruction(true_idces, probas, idx_to_char, string_type='smiles'):
    """
    Input : 
        true_idces : shape (N, seq_len)
        probas : shape (N, voc_size, seq_len)
        idx_to_char : dict with idx to char mapping 
        string-type : smiles or selfies 
        
        Argmax on probas array (dim 1) to find most likely character indices 
    """
    probas = probas.to('cpu').numpy()
    true_idces = true_idces.cpu().numpy()
    N, voc_size, seq_len = probas.shape
    out_idces = np.argmax(probas, axis=1)  # get char_indices
    in_smiles, out_smiles = [], []
    for i in range(N):
        out_smiles.append(''.join([idx_to_char[idx] for idx in out_idces[i]]))
        in_smiles.append(''.join([idx_to_char[idx] for idx in true_idces[i]]))
    d = {'input smiles': in_smiles,
         'output smiles': out_smiles}
    
    if string_type =='smiles':
        df = pd.DataFrame.from_dict(d)
        valid = [Chem.MolFromSmiles(o.rstrip('\n')) for o in out_smiles]
        valid = [int(m != None) for m in valid]
        frac_valid = np.mean(valid)
        return df, frac_valid
    else:
        for i in range(3): #printing only 3 samples 
            print('True sample selfies : ')
            print(in_smiles[i])
            print('Reconstruction : ')
            print(out_smiles[i])
        return 0,0


def log_smiles_from_indices(true_idces, out_idces, idx_to_char):
    """
    Input : 
        true_idces : shape (N, seq_len)
        out_idces : shape (N, seq_len)
        idx_to_char : dict with idx to char mapping 
    """
    N, seq_len = out_idces.shape
    if (type(true_idces) == np.ndarray):
        print('shape of true indices array: ', true_idces.shape)
        input_provided = True
    else:
        print('No input smiles given, random sampling from latent space ?')
        input_provided = False
    print('shape of output indices array: ', out_idces.shape)
    in_smiles, out_smiles = [], []
    identical = 0
    for i in range(N):
        if (input_provided):
            out_smiles.append(''.join([idx_to_char[idx] for idx in out_idces[i]]))
            in_smiles.append(''.join([idx_to_char[idx] for idx in true_idces[i]]))
            if (in_smiles == out_smiles):
                identical += 1
        else:  # Consider only valid smiles
            out = ''.join([idx_to_char[idx] for idx in out_idces[i]])
            if (Chem.MolFromSmiles(out.rstrip('\n')) != None):
                out_smiles.append(out)
    if (input_provided):
        d = {'input smiles': in_smiles,
             'output smiles': out_smiles}
        valid = [Chem.MolFromSmiles(o.rstrip('\n')) for o in out_smiles]
        valid = [int(m != None) for m in valid]
        frac_valid = np.mean(valid)
        frac_id = identical / N
    else:
        d = {'output smiles': out_smiles}
        frac_valid = len(out_smiles) / N
        frac_id = 0  # not applicable
    df = pd.DataFrame.from_dict(d)

    return df, frac_valid, frac_id


def i2s(idces, idx_to_char):
    # list of indices to sequence of characters (=smiles)
    return ''.join([idx_to_char[idx] for idx in idces])

