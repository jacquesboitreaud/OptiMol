# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE evaluation (RGCN encoder, GRU decoder, beam search decoding). 


"""
import sys
import torch
import dgl
import pickle
import torch.utils.data
from torch import nn, optim
import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if (__name__ == "__main__"):
    sys.path.append("./dataloading")
    from model import Model, Loss, RecLoss
    from molDataset import molDataset, Loader
    from utils import *
    from plot_tools import pca_plot, pca_plot_true_affs
    
    # config
    batch_size = 100
    SAVE_FILENAME='./saved_model_w/g2s.pth'
    model_path= 'saved_model_w/g2s.pth'
    
    properties = ['QED','logP','molWt','maxCharge','minCharge','valence','TPSA','HBA','HBD']
    targets = ['t1','t2']

    #Load train set and test set
    loaders = Loader(csv_path='data/validation_2targets.csv',
                     n_mols=11000,
                     num_workers=0, 
                     batch_size=batch_size, 
                     shuffled= False,
                     props = properties,
                     targets=targets,
                     test_only=True)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    
    _, _, test_loader = loaders.get_data()
    
    #Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parallel=False
    
    params = pickle.load(open('saved_model_w/params.pickle','rb'))
    model = Model(**params).to(device)
    model.load_state_dict(torch.load(model_path))
    
    if (parallel): #torch.cuda.device_count() > 1 and
        print("Parallel model using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    #Print model summary
    print(model)
    map = ('cpu' if device == 'cpu' else None)
    torch.manual_seed(1)
    
    # Validation pass
    model.eval()
    t_rec, t_mse = 0,0
    z_all, affs_all = np.zeros((loaders.dataset.n,model.l_size)), np.zeros((loaders.dataset.n,len(targets)))
    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
            
            p_target=p_target.to(device).view(-1,len(properties)) # Graph-level target : (batch_size,)
            a_target=a_target.to(device).view(-1,len(targets))
            smiles=smiles.to(device)
            graph=send_graph_to_device(graph,device)
            
            # Latent representations
            z = model.encode(graph, mean_only=True) #z_shape = N * l_size
            
            """
            # Decoding to smiles (beam search) and predicted props
            beam_output = model.decode_beam(z,k=3)
            props, aff = model.props(z), model.affs(z)
            # Only valid out molecules 
            mols = log_from_beam(beam_output,loaders.dataset.index_to_char)
            """
            
            #TODO:  Concat all z and affs 
            
            z=z.cpu().numpy()
            z_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=z
            affs = a_target.cpu().numpy()
            affs_all[batch_idx*batch_size:(batch_idx+1)*batch_size]=affs
            
            
        # Plot 
        bool1 = [int(a[0]!=0) for a in affs_all]
        bool2 = [int(a[1]!=0) for a in affs_all]
        bit = np.array(bool2)+ 10*np.array(bool1) # bit affinities 
        bit = [str(i) for i in bit]
        mapping = {'0':'dd','1':'da','10':'ad','11':'aa'}
        bit = [mapping[b] for b in bit]
        bit=pd.Series(bit,index=np.arange(len(bit)))
        
        pca_plot_true_affs(z_all,bit)
            
            
            
            
                     

            
            
            
            
            
            
        #reconstruction_dataframe = log_smiles(smiles, out, loaders.dataset.index_to_char)
        #print(reconstruction_dataframe)
        