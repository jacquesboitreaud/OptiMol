# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:41:22 2020

@author: jacqu

Loads trained model into memory 
"""

import torch
import numpy as np

import pickle
import torch.utils.data


import sys
sys.path.append('../')
from model import Model
    
from utils import *
    
def load_trained_model(model_path):
    # Loads model and returns model instance + model's device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    model = Model(**params).to(device)
    model.load_state_dict(torch.load(model_path))
        
    #Print model summary
    print(model)
    
    return model, device

def embed(model, device, loader, df):
    # Gets latent embeddings of molecules in df 
    
    loader.dataset.pass_dataset(df)
    _, _, test_loader = loader.get_data()
    batch_size=loader.batch_size
    
    # Latent embeddings
    z_all = np.zeros((loader.dataset.n,model.l_size))
    
    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
            
            graph=send_graph_to_device(graph,device)
        
            z = model.encode(graph, mean_only=True) #z_shape = N * l_size
            z=z.cpu().numpy()
            z_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=z
            
    return z_all
