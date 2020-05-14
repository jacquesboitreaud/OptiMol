# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Optimize affinity with bayesian optimization. 

Following tutorial at https://botorch.org/tutorials/vae_mnist

****Logging and list of already scored molecules to implement 



"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd 

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import pickle
import torch.utils.data

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from selfies import decoder
from rdkit import Chem

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler



if __name__ == "__main__":
    
    from dataloaders.molDataset import Loader
    from model import Model, model_from_json
    from utils import *
    from dgl_utils import * 
    from bo_utils import get_fitted_model
    from docking.docking_debug import dock, set_path

    parser = argparse.ArgumentParser()
    
    parser.add_argument( '--bo_name', help="Name for BO results subdir ",
                        default='diverse')
    
    parser.add_argument( '--name', help="saved model weights fname. Located in saved_models subdir",
                        default='inference_default')
    parser.add_argument('-n', "--n_steps", help="Nbr of optim steps", type=int, default=20)
    parser.add_argument('-q', "--n_queries", help="Nbr of queries per step", type=int, default=50)
    parser.add_argument('-o', '--objective', default='aff_pred') # 'qed', 'aff', 'aff_pred'
    
    # initial samples to use 
    parser.add_argument('--init_samples', default='2k_diverse_samples.csv') # samples to start with // random or excape data
    parser.add_argument('--n_init', type = int ,  default=2000) # Number of samples to start with 
    
    # docking specific params 
    parser.add_argument('-e', "--ex", help="Docking exhaustiveness (vina)", type=int, default=16) 
    parser.add_argument('-s', "--server", help="COmputer used, to set paths for vina", type=str, default='rup')
    parser.add_argument( '--load', default='drd3_scores.pickle') # Pickle file with dict of already docked molecules // keyed by kekuleSMiles
    
    parser.add_argument('-v', "--verbose", help="print new scores at each iter", action = 'store_true', default=False)
    args, _ = parser.parse_known_args()

    # ==============
    
    with open(os.path.join(script_dir,'..', 'docking', args.load), 'rb') as f:
        load_dict = pickle.load(f)
    print(f'Preloaded {len(load_dict)} docking scores')
    
    soft_mkdir('bo_results')
    soft_mkdir(os.path.join('bo_results',args.bo_name))

    vocab = 'selfies'
    # Loader for initial sample
    loader = Loader(props=[], 
                    targets=[], 
                    csv_path = None,
                    vocab=vocab, 
                    num_workers = 0,
                    test_only=True)

    # Load model (on gpu if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # the model device 
    model = model_from_json(args.name)
    model.to(device)
    model.eval()
    
    # Search space 
    d = model.l_size
    dtype = torch.float
    bounds = torch.tensor([[-3.0] * d, [3.0] * d], device='cpu', dtype=dtype)
    BO_BATCH_SIZE = args.n_queries
    N_STEPS = args.n_steps
    MC_SAMPLES = 2000
    
    seed=1
    torch.manual_seed(seed)
    best_observed = []
    state_dict = None
    
    # Generate initial data 
    df = pd.read_csv(os.path.join(script_dir,'..','data',args.init_samples), nrows = args.n_init) # n_init Initial samples 
    
    loader.graph_only=True
    train_z = torch.tensor(model.embed( loader, df)) # z has shape (N_molecules, latent_size)
    
    if args.objective == 'qed':
        scores_init = torch.tensor([Chem.QED.qed(Chem.MolFromSmiles(s)) for s in df.smiles]).view(-1,1).to(device)
    elif args.objective == 'aff_pred':
        with torch.no_grad():
            scores_init = -1* model.affs(train_z.to(device)).cpu() # careful, maximize -aff <=> minimize binding energy (negative value)
    elif args.objective == 'aff' : 
        PYTHONSH, VINA = set_path(args.server)
        scores_init = -1* torch.tensor(df.drd3).view(-1,1).cpu() # careful, maximize -aff <=> minimize binding energy (negative value)
    
    # Tracing results
    sc_dict = {}
    best_value = torch.max(scores_init).item()
    best_observed.append(best_value)
    train_obj = scores_init
    train_smiles = list(df.smiles)
    print(f'-> Best value observed in initial samples : {best_value}')
    
    # Acquisition function 
    def optimize_acqf_and_get_observation(acq_func, device):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
        
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([
                torch.zeros(d, dtype=dtype, device='cpu'), 
                torch.ones(d, dtype=dtype, device='cpu'),
            ]),
            q=BO_BATCH_SIZE,
            num_restarts=10,
            raw_samples=200,
        )
    
        # observe new values 
        new_z = unnormalize(candidates.detach(), bounds=bounds).to(device)
        
        # Decode z into smiles
        with torch.no_grad():
            gen_seq = model.decode(new_z.to(device))
            smiles = model.probas_to_smiles(gen_seq)
        if vocab=='selfies' :
            k = []
            for s in smiles :
                s = decoder(s)
                m = Chem.MolFromSmiles(s)
                Chem.Kekulize(m)
                s= Chem.MolToSmiles(m, kekuleSmiles = True)
                k.append(s)
            smiles = k 
            
            
        
        if BO_BATCH_SIZE > 1 : # query a batch of smiles 
            
            if args.objective == 'aff':
                new_scores = torch.zeros((BO_BATCH_SIZE,1), dtype=torch.float)
                for i in range(len(smiles)):
                    sc = dock(smiles[i], i, PYTHONSH, VINA, exhaustiveness = args.ex, load = False )
                    new_scores[i,0]=sc
                    # Update dict with scores 
                    if smiles[i] not in load_dict and sc < 0 :
                        load_dict[smiles[i]]= sc
                    
                new_scores = -1* new_scores
                
                
                
            elif args.objective == 'aff_pred':
                with torch.no_grad():
                    new_scores = -1* model.affs(new_z).cpu() # shape N*1
                    for i in range(len(smiles)):
                        if not isValid(smiles[i]):
                            new_scores[i,0]=0.0 # assigning 0 score to invalid molecules for realistic oracle 
                
            elif args.objective =='qed':
                mols = [Chem.MolFromSmiles(s) for s in smiles ]
                new_scores = torch.zeros(len(smiles), dtype = torch.float)
                for i,m in enumerate(mols):
                    if m!=None:
                        new_scores[i]=Chem.QED.qed(m)
                new_scores = new_scores.unsqueeze(-1)  # new scores must be (N*1)
                
            return smiles, new_z, new_scores
        
        else: #query an individual smiles 
            raise NotImplementedError
    
    
    # ========================================================================
    # run N_BATCH rounds of BayesOpt after the initial random batch
    # ========================================================================
    print('-> Invalid molecules get score 0.0')
    samples_score = [] # avg score of new samples at each step 
    
    for iteration in range(N_STEPS):    
        print(f'Iter [{iteration}/{N_STEPS}]')
        # fit the model
        GP_model = get_fitted_model(
            normalize(train_z, bounds=bounds), 
            standardize(train_obj), 
            state_dict=state_dict,
        )
        
        # define the qNEI acquisition module using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
        qEI = qExpectedImprovement(model=GP_model, sampler=qmc_sampler, best_f=standardize(train_obj).max())
    
        # optimize and get new observation
        new_smiles, new_z, new_score = optimize_acqf_and_get_observation(qEI, device)
        
        # save acquired scores for next time 
        if(args.verbose):
            print(' oracle outputs:')
            print(new_score.numpy())
        sc_dict[iteration]=new_score.numpy()
    
        # update training points
        
        train_smiles+= new_smiles 
        train_z = torch.cat((train_z, new_z.cpu()), dim=0)
        train_obj = torch.cat((train_obj, new_score), dim=0)
        state_dict = GP_model.state_dict()
    
        # update progress
        avg_score = torch.mean(new_score).item()
        best_value, idx = torch.max(train_obj,0)
        samples_score.append(avg_score)
        best_observed.append(best_value.item())
        idx = idx.item()
        best_smiles = train_smiles[idx]
        
        print(f'current best mol: {best_smiles}, with oracle score {best_value.item()}')
        print(f'average score of fresh samples at iter {iteration}: {avg_score}')
        print("\n")
        
        # Save epoch samples 
        with open(os.path.join('bo_results',args.bo_name,'sample_scores.pickle'), 'wb') as f :
            pickle.dump(sc_dict, f)
        
        # Update file with top 100 samples discovered 
        train_obj_flat=train_obj.numpy().flatten()
        idces = np.argsort(train_obj_flat)
        idces=idces[-100:] # top 100
        with open(os.path.join('bo_results',args.bo_name,f'top_samples_{iteration}.txt'), 'w') as f :
            for i in idces :
                f.write(train_smiles[i] +',  ' + str(train_obj_flat[i].item())+'\n' )
        print('wrote top samples and scores to txt. ')
        
        # Save updated dict with docking scores 
        if args.objective =='aff':
            with open(os.path.join(script_dir,'..','docking', args.load), 'wb') as f:
                pickle.dump(load_dict,f)
        
        
    

    
    
        
        
