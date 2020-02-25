# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:45:02 2019

@author: jacqu

Gradient descent optimization of objective target function in latent space.
Testing optimization of DRD3 binding for multiple seeds, with a loop. 

Writes output to csv 

Change the objective function in eps(props, aff)
Change learning rate accordingly

TODO : finetune this for better results + more rigorous process. 

"""

import os
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import torch
import numpy as np
from numpy.linalg import norm
import pandas as pd

from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors

import sys
import matplotlib.pyplot as plt
import seaborn as sns

from model import Model

if(__name__=='__main__'):
    from data_processing.rdkit_to_nx import smiles_to_nx
    from dataloaders.molDataset import molDataset
    from utils import *

    # Params 
    
    N = 50 # Number of seeds 
    beam = False # use beam search (! slow, ~ 2-3 secs per molecule)
    prints = False # plt show molecules + print predicted and true props at each step 
    
    seeds_df = pd.read_csv('../data/moses_affs_test.csv') # dataframe with seeds for optim start 
    seeds_df= seeds_df.sample(N).reset_index(drop=True)
    seeds= list(seeds_df['can'])
    
    # PB / Without beam search, most steps lead to invalid smiles decoding 
    N_steps=50 # N steps per molecule 
    
    size = (120, 120) # mol figsize  
    
    def eps(props, aff):
        """ objective function to MINIMIZE in latent space """
        # props = [QED, logP, molWt]
        # aff = [drd3_aff] is negative !!!!
        qed, logP, molWt = props
        logP_tar = None # eventually set target logp 
        
        eps = - 10*qed + aff[0]
        
        if(logP_tar != None):
            eps += (logP- logP_tar)**2
        
        return eps
    
    molecules =[]
    starts_a, starts_q = [],[]
    data = molDataset(maps_path='../map_files/',
                      csv_path=None)
    seed_results = []
    for k,s_seed in enumerate(seeds):
        lr=0.1
        if(k<2):
            next
        print('Optimizing seed n° ', k)
        # Begin optimization with seed compound   

        m0= Chem.MolFromSmiles(s_seed)
        starts_q.append(Chem.QED.default(m0))
        starts_a.append(seeds_df.loc[k,'drd3'])
        """
        Draw.MolToMPL(m0, size=size)
        plt.show(block=False)
        plt.pause(0.1)
        """
    
        # Pass to loader 
        data.pass_smiles_list([s_seed])
        graph,_,_,_ = data.__getitem__(0)
        send_graph_to_device(graph,model.device)
        
        # pass to model
        z = model.encode(graph, mean_only=True)
        z=z.unsqueeze(dim=0)

    
        # ================= Optimization process ================================
        best = (0,0,0)
        
        for i in range(N_steps):
            # Objective function 
            epsilon= eps(model.props(z)[0], model.affs(z)[0])
            g=torch.autograd.grad(epsilon,z)
            with torch.no_grad(): # Gradient descent, don't track
                z = z - lr*g[0]
                if(i%10==0):
                    lr=0.9*lr
                
                if(beam):
                    out=model.decode_beam(z)
                    smi = model.beam_out_to_smiles(out)[0]
                else:
                    out=model.decode(z)
                    smi = model.probas_to_smiles(out)[0]
                
                print(smi)
                m=Chem.MolFromSmiles(smi)
                logP=0
                if(i==0):
                    prev_s=smi
                
                if(m!=None and smi!=prev_s):
                    plt.show(block=False)
                    plt.pause(0.1)
                
                    prev_s=smi
                    logP= Chem.Crippen.MolLogP(m)
                    qed = Chem.QED.default(m)
                    pred_aff = model.affs(z)[0,0].item()
                    if(prints):
                        Draw.MolToMPL(m, size=size)
                        print(f'predicted logP: {model.props(z)[0,1].item():.2f}, true: {logP:.2f}')
                        print(f'predicted QED: {model.props(z)[0,0].item():.2f}, true: {qed:.2f}')
                        print(f'predicted aff: {pred_aff:.2f}')
                    
                    if(pred_aff<best[1]):
                        best = (smi, pred_aff, qed)
                else:
                    if(prints):
                        print(f'predicted logP / QED : {model.props(z)[0,1].item():.2f} / {model.props(z)[0,0].item():.2f}, invalid smiles')
            z.requires_grad =True
        
        # End of process for this seed
        print('seed n° ', k, best)
        seed_results.append(best)
        
    d={'seed':list(np.arange(len(seed_results))),
       'can':[t[0] for t in seed_results],
       'pred_aff':[t[1] for t in seed_results],
       'QED':[t[2] for t in seed_results ],
       'start_a': starts_a,
       'start_qed': starts_q}
    
    df=pd.DataFrame.from_dict(d)
        
    df.to_csv('aff_optim_results.csv')