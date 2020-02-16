# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Sampling molecules in latent space 
"""

import torch
from rdkit.Chem import Draw
import sys
    
def sample_around_mol(model, smi, d, beam_search=False, attempts = 1,
                      props=False, aff=False):
    # Samples molecules at a distance around molecules in dataframe 'can' column
    data=molDataset()
    data.pass_smiles_list(smi)
    
    g_dgl,_, _,_ = data.__getitem__(0)
    send_graph_to_device(g_dgl,model.device)
    
    # pass to model
    with torch.no_grad():
        gen_seq, _,_ = model.sample_around_mol(g_dgl, dist=d, beam_search=beam_search, 
                                               attempts=attempts,props=props,aff=aff) # props & affs returned in _
        
    # Sequence to smiles 
    if(not beam_search):
        smiles = model.probas_to_smiles(gen_seq)
        
    else:
        smiles = model.beam_out_to_smiles(gen_seq)
        
    return smiles

def sample_prior():
    # Sample from uniform prior 
    r=model.sample_z_prior(100)
    # Decode 
    out = model.decode(r)
    v, indices = torch.max(out, dim=1)
    indices = indices.cpu().numpy()
    return indices

if(__name__=='__main__'):
    
    sys.path.append('../')
    sys.path.append('data_processing')
    sys.path.append('eval')
    from molDataset import molDataset
    from model_utils import load_trained_model
    from utils import *
    
    
    # Sampling around given molecule 
    use_beam = False
    smi = ['Oc4ccc3C(Cc1ccccc1)N(c2ccccc2)CCc3c4']
    m0 = Chem.MolFromSmiles(smi[0])
    Draw.MolToMPL(m0, size=(120, 120))
    plt.show(block=False)
    
    out_smi = sample_around_mol(model, smi, d=0, beam_search=use_beam, attempts = 200)

    mols= [Chem.MolFromSmiles(s) for s in out_smi]
    mols = [m for m in mols if m!=None]
    
    size = (120, 120)
    for m in mols:
        Draw.MolToMPL(m, size=size)
        if(m==m0):
            break
        plt.show(block=False)
        plt.pause(0.1)