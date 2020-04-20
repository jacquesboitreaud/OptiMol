# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:45:02 2019

@author: jacqu

Gradient descent optimization of objective target function in latent space.

Starts from 1 seed compound or random point in latent space (sampled from prior N(0,1))

TODO : 
    - add tanimoto similarity to seed compound
"""

import os
import sys
import pickle 
import torch
import numpy as np
from numpy.linalg import norm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw, QED, Crippen, Descriptors
from rdkit.Chem import MACCSkeys
import matplotlib.pyplot as plt



if(__name__=='__main__'):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))
    from data_processing.rdkit_to_nx import smiles_to_nx
    from dataloaders.molDataset import molDataset
    from utils import *
    from model import Model
    
    # Set to true if providing a seed compound to start optimization. Otherwise, random start in the latent space
    model_path = 'saved_model_w/g2s_iter_440000.pth'
    seed= False
    N_steps=100
    lr=0.1
    size = (120, 120) # plotting 
    early_stopping_QED = 0.940
    
    def eps(props, aff):
        # props = [QED, logP, molWt]
        # aff = [drd3_aff] is negative !!!!
        QED, logP, molWt = props
        #return  (QED-1)**2 + aff[0]
        return  -10*QED
    
    molecules =[]
    data = molDataset(csv_path = None, 
                      maps_path='../map_files/',
                      vocab = 'selfies', 
                      build_alphabet = False, 
                      props = ['QED', 'logP', 'molWt'],
                      targets = [])
    
    # Load model 
    # Load model (on gpu if available)
    params = pickle.load(open(os.path.join(script_dir,'..','saved_model_w/model_params.pickle'), 'rb'))  # model hparams
    model = Model(**params)
    model.load(os.path.join(script_dir, '..', model_path))
    model.eval()
    
    # ================== Starting from a seed compound ======================
    if(seed):
        # Begin optimization with seed compound : a DUD decoy   
        s_seed='O=C(NC1=CCCC1=O)NC1=CCN(c2ncnc3ccccc23)CC1'
        m0= Chem.MolFromSmiles(s_seed)
        fp0 = MACCSkeys.GenMACCSKeys(m0) # fingerprint (MACCS)
        
        Draw.MolToMPL(m0, size=size)
        plt.show(block=False)
        plt.pause(0.1)
    
        # Pass to loader 
        data.pass_smiles_list([s_seed])
        graph,_,_,_ = data.__getitem__(0)
        send_graph_to_device(graph,model.device)
        
        # pass to model
        z = model.encode(graph, mean_only=True)
        z=z.unsqueeze(dim=0)
    else:
        # ======================= Starting from random point ===============
        print("Sampling random latent vector")
        z = model.sample_z_prior(1)
        z.requires_grad=True
    
    # ================= Optimization process ================================
    for i in range(N_steps):
        # Objective function 
        epsilon= eps(model.props(z)[0], model.affs(z)[0])
        g=torch.autograd.grad(epsilon,z)
        with torch.no_grad(): # Gradient descent, don't track
            z = z - lr*g[0]
            if(i%20==0):
                lr=0.9*lr
            
            out=model.decode(z)
            smi = model.probas_to_smiles(out)[0]
            if data.language == 'selfies':
                smi = selfies.decoder(smi)
            
            #out=model.decode_beam(z)
            #smi = model.beam_out_to_smiles(out)[0]
            
            print(smi)
            m=Chem.MolFromSmiles(smi)
            logP=0
            if(i==0):
                prev_s=smi
            
            if(m!=None and i%10==0): # print every 10 steps 
                Draw.MolToMPL(m, size=size)
                plt.show(block=False)
                plt.pause(0.1)
            
                prev_s=smi
                logP= Chem.Crippen.MolLogP(m)
                qed = Chem.QED.default(m)
                
                print(f'predicted logP: {model.props(z)[0,1].item():.2f}, true: {logP:.2f}')
                print(f'predicted QED: {model.props(z)[0,0].item():.4f}, true: {qed:.4f}')
                #print(f'predicted aff: {model.affs(z)[0,0].item():.2f}')
                if(qed >= early_stopping_QED):
                    if(seed):
                        fp = MACCSkeys.GenMACCSKeys(m)
                        tanimoto = DataStructs.FingerprintSimilarity(fp0,fp, metric=DataStructs.TanimotoSimilarity)
                        print('Tanimoto similarity to seed compound: ', tanimoto)
                    break
            else:
                print(f'predicted logP / QED : {model.props(z)[0,1].item():.2f} / {model.props(z)[0,0].item():.2f}, invalid smiles')
        z.requires_grad =True