# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:45:03 2020

@author: jacqu

Pass unseen molecules to trained model to estimate the validation/inference metric 
"""

import argparse
import sys, os
import torch
import numpy as np

import pandas as pd
import torch.utils.data
from torch import nn, optim


script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(script_dir)

from utils import ModelDumper, disable_rdkit_logging, setup, log_reconstruction
from dgl_utils import send_graph_to_device
from model_zinc import Model, model_from_json
from loss_func import VAELoss, weightedPropsLoss, affsRegLoss, affsClassifLoss
from dataloaders.molDataset import molDataset, Loader

from selfies import decoder

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem

from time import time

parser = argparse.ArgumentParser()

parser.add_argument('--train', help="path to training dataframe", type=str, default='data/shuffled_whole_zinc.csv')

# Alphabets params 
parser.add_argument('--decode', type=str, default='selfies')  # language used : 'smiles' or 'selfies'
parser.add_argument('--alphabet_name', type=str,
                    default='zinc_alphabets.json')  # name of alphabets json file, in map_files dir
parser.add_argument('--build_alphabet', action='store_true')  # Ignore json alphabet and build from scratch

# If we start from a pretrained model :
parser.add_argument('--load_name', type=str, default='default')  # name of model to load from
parser.add_argument('--load_iter', type=int, default=0)  # resume training at optimize step n°

# Model architecture
parser.add_argument('--decoder_type', type=str, default='GRU')  # name of model to load from
parser.add_argument('--n_gcn_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
parser.add_argument('--n_gru_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
parser.add_argument('--gcn_dropout', type=float, default=0.2)
parser.add_argument('--gcn_hdim', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=56) # jtvae uses 56
parser.add_argument('--gru_hdim', type=int, default=450)
parser.add_argument('--gru_dropout', type=float, default=0.2)

parser.add_argument('--use_batchNorm', action='store_true') # default uses batchnorm tobe coherent with before 

# Training schedule params :

parser.add_argument('--batch_size', type=int, default=64)

# Multitask :
parser.add_argument('--no_props', action='store_false')  # No multitask props

# Other : 
parser.add_argument('--gpu_id', type=int, default=0)  # run model on cuda:{id} if multiple gpus on server
parser.add_argument('--processes', type=int, default=4)  # num workers

if __name__ == '__main__':

    args, _ = parser.parse_known_args()
    
    unseen_samp = pd.read_csv('data/10k_unseen_zinc.csv') # 10k zinc mols between 30e6 and 30e6 + 10 000
        
    if args.n_gru_layers ==4 :
        raise NotImplementedError
    
    
    use_props, use_affs = True, False
    if args.no_props:
        use_props = False
    
    # Multitasking : properties and affinities should be in input dataset
    if use_props:
        properties = ['QED', 'logP', 'molWt']
    else:
        properties = []
    props_weights = [1e3, 1e2, 1]
    
    
    targets = []
    
    disable_rdkit_logging()  # function from utils to disable rdkit logs
    
    # Empty loader object 
    loaders = Loader(maps_path='map_files/',
                     csv_path=None,
                     vocab=args.decode,
                     build_alphabet=args.build_alphabet,
                     alphabet_name=args.alphabet_name,
                     n_mols=-1,
                     num_workers=args.processes,
                     batch_size=args.batch_size,
                     props=properties,
                     targets=targets,
                     redo_selfies = False) # recompute selfies in the dataloader instead of using dataframe value 
    
    # Model & hparams
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu' # multiple GPU in argparse
    model = model_from_json(name=args.load_name).to(device)
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    
    # Pass valid mols 
    
    loaders.dataset.pass_dataset(unseen_samp, graph_only=False)
    train_loader, _, test_loader = loaders.get_data()
    
    # Validation pass : No teacher forcing for decoding (sampling mode)
    model.eval()
    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
    
            smiles = smiles.to(device)
            graph = send_graph_to_device(graph, device)
    
            if use_props:
                p_target = p_target.to(device).view(-1, len(properties))
    
            mu, logv, z, out_smi, out_p = model(graph, smiles, tf=1.0, mean_only=True) # no gaussian sampling here 
    
            # Compute loss : change according to multitask
    
            rec, kl = VAELoss(out_smi, smiles, mu, logv)
            if not use_props:  # VAE only
                pmse = torch.tensor(0)
            elif use_props :
                pmse = weightedPropsLoss(p_target, out_p, props_weights)
    
            # Correctly reconstructed characters in first validation batch
            _, out_chars = torch.max(out_smi.detach(), dim=1)
            differences = 1. - torch.abs(out_chars - smiles)
            differences = torch.clamp(differences, min=0., max=1.).double()
            quality = 100. * torch.mean(differences)
            quality = quality.detach().cpu()
            print('fraction of correct characters at reconstruction // validation : ', quality.item())
            
            # Tanimoto to input mols : 
            inputs = model.indices_to_smiles(smiles)
            truth = [decoder(seq) for seq in inputs]
            out = model.indices_to_smiles(out_chars)
            dec = [decoder(seq) for seq in out]
            
            sims= []
            for s_i, s_o in zip(truth,dec):
                m_i = Chem.MolFromSmiles(s_i)
                m_o = Chem.MolFromSmiles(s_o)
                
                fp_i = AllChem.GetMorganFingerprintAsBitVect(m_i, 3,
                                                   nBits=1024)  # careful radius = 3 equivalent to ECFP 6 (diameter = 6, radius = 3)
                fp_o = AllChem.GetMorganFingerprintAsBitVect(m_o, 3,
                                                   nBits=1024)  
                
                sims.append( DataStructs.FingerprintSimilarity(fp_i, fp_o))
            print('Avg tanimoto similarity btw input and output mols: ', np.mean(np.array(sims)))
                
            