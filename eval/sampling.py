# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space 
"""

import torch
from rdkit.Chem import Draw

from model_utils import load_trained_model

def sample_z_prior( batch_size, l_size):
    """Sampling z ~ p(z) = N(0, I)
    :param n_batch: number of batches
    :return: (n_batch, d_z) of floats, sample of latent z
    """

    return torch.randn(batch_size, l_size,
               device='cuda')


if(__name__=='__main__'):
    
    # Sample from uniform prior 
    r=sample_z_prior(100,64)
    
    # Decode 
    out = model.decode(r)
    v, indices = torch.max(out, dim=1)
    indices = indices.cpu().numpy()
    sampling_df, frac_valid,_ = log_smiles_from_indices(None, indices, 
                                                  loaders.dataset.index_to_char)
    
    # Uncomment for beam search decoding 
    # beam_output = model.decode_beam(z,k=3, cutoff_mols=10)
    # mols, _ = log_from_beam(beam_output,loaders.dataset.index_to_char)
    
    props, affs = model.props(r).detach().cpu().numpy(), model.affs(r).detach().cpu().numpy()
    mols= [Chem.MolFromSmiles(s) for s in list(sampling_df['output smiles'])]
    
    fig = Draw.MolsToGridImage(mols)
    
    #TODO; mask valid properties and affinities 