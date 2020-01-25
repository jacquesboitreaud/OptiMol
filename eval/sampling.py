# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space 
"""

import torch
from rdkit.Chem import Draw
import sys
    
def sample_around_mol(model, smi, d, beam_search=False, attempts = 1):
    # Samples molecules at a distance around molecules in dataframe 'can' column
    data=molDataset()
    data.pass_smiles_list(smi)
    
    g_dgl,_, _,_ = data.__getitem__(0)
    send_graph_to_device(g_dgl,model.device)
    
    # pass to model
    with torch.no_grad():
        gen_seq = model.sample_around_mol(g_dgl, dist=d, beam_search=beam_search, attempts=attempts)
        
    return gen_seq


if(__name__=='__main__'):
    
    sys.path.append('../')
    sys.path.append('data_processing')
    sys.path.append('eval')
    from molDataset import molDataset
    from model_utils import load_trained_model
    from utils import *
    
    """
    # Sample from uniform prior 
    r=model.sample_z_prior(100)
    # Decode 
    out = model.decode(r)
    v, indices = torch.max(out, dim=1)
    indices = indices.cpu().numpy()
    sampling_df, frac_valid,_ = log_smiles_from_indices(None, indices, 
                                                  loaders.dataset.index_to_char)
    
    # Uncomment for beam search decoding 
    # beam_output = model.decode_beam(z,k=3, cutoff_mols=10)
    # mols = log_from_beam(beam_output,loaders.dataset.index_to_char)
    
    props, affs = model.props(r).detach().cpu().numpy(), model.affs(r).detach().cpu().numpy()
    mols= [Chem.MolFromSmiles(s) for s in list(sampling_df['output smiles'])]
    fig = Draw.MolsToGridImage(mols)
    """
    
    # Sampling around given molecule 
    use_beam = True
    smi = ['Oc4ccc3C(Cc1ccccc1)N(c2ccccc2)CCc3c4']
    out = sample_around_mol(model, smi, d=4, beam_search=use_beam, attempts = 10)
    if(not use_beam):
        v, indices = torch.max(out, dim=1)
        indices = indices.cpu().numpy()
        sampling_df, frac_valid,_ = log_smiles_from_indices(None, indices, 
                                                  loaders.dataset.index_to_char)
        mols= [Chem.MolFromSmiles(s) for s in list(sampling_df['output smiles'])]
    else:
        smiles = log_from_beam(out,loaders.dataset.index_to_char)
        mols= [Chem.MolFromSmiles(s) for s in smiles]
        mols = [m for m in mols if m!=None]
    
    fig = Draw.MolsToGridImage(mols)