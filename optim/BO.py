# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Optimize affinity with bayesian optimization. 

Following tutorial at https://botorch.org/tutorials/vae_mnist



"""
import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd 
import csv 
from multiprocessing import Pool

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import pickle
import torch.utils.data

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

from selfies import decoder
from rdkit import Chem
from rdkit.Chem import AllChem

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from sklearn.svm import SVC

from datetime import datetime
from time import time 


if __name__ == "__main__":
    
    from dataloaders.molDataset import Loader
    from model import Model, model_from_json
    from utils import *
    from dgl_utils import send_graph_to_device
    from BO_utils import get_fitted_model
    from docking.docking import dock, set_path
    from data_processing.comp_metrics import cLogP, cQED

    parser = argparse.ArgumentParser()
    
    parser.add_argument( '--bo_name', help="Name for BO results subdir ",
                        default='clopg_big')
    
    parser.add_argument('-o', '--objective', default='clogp') # 'cqed', 'clogp' 'aff', 'aff_pred' or 'qsar'
    
    parser.add_argument( '--name', help="saved model weights fname. Located in saved_models subdir",
                        default='250k')
    
    parser.add_argument('-n', "--n_steps", help="Nbr of optim steps", type=int, default=20)
    parser.add_argument('-q', "--n_queries", help="Nbr of queries per step", type=int, default=500)
    
    # initial samples to use 
    parser.add_argument('--init_samples', default='250k_zinc.csv') # samples to start with // random or excape data
    parser.add_argument('--n_init', type = int ,  default=10000) # Number of samples to start with 
    
    # docking specific params 
    parser.add_argument('-e', "--ex", help="Docking exhaustiveness (vina)", type=int, default=16) 
    parser.add_argument('-s', "--server", help="Computer used, to set paths for vina", type=str, default='rup')
    parser.add_argument( '--load', default='drd3_scores.pickle') # Pickle file with dict of already docked molecules // keyed by kekuleSMiles
    
    # qsar model path 
    parser.add_argument("--qsar", help="Path to pickle svm, from repo root", type=str, default='results/saved_models/qsar_svm.pickle')
    
    
    parser.add_argument('-v', "--verbose", help="print new scores at each iter", action = 'store_true')
    args, _ = parser.parse_known_args()

    # ==============
    
    n_runs = 1
    
    for seed in range(1,n_runs+1):
        
        print(f'Start sim {seed}')
        
        if args.objective == 'aff':
            with open(os.path.join(script_dir,'..', 'docking', args.load), 'rb') as f:
                load_dict = pickle.load(f)
            print(f'Preloaded {len(load_dict)} docking scores')
        
        #time_id = datetime.now()
        #time_id = time_id.strftime("_%d_%H%M")
    
        # ALphabets and language 
        vocab = 'selfies'
        if '250k' in args.name:
            alphabet = '250k_alphabets.json'
        elif 'zinc' in args.name:
            alphabet = 'zinc_alphabets.json'
        else:
            alphabet = 'moses_alphabets.json'
            
        # Loader for initial sample
        loader = Loader(props=[], 
                        targets=[], 
                        csv_path = None,
                        alphabet_name = alphabet,
                        vocab=vocab, 
                        num_workers = 0,
                        test_only=True)
    
        # Load model (on gpu if available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # the model device 
        gp_device =  'cpu' #'cuda' if torch.cuda.is_available() else 'cpu' # gaussian process device 
        model = model_from_json(args.name)
        model.to(device)
        model.eval()
        
        # Search space 
        d = model.l_size
        dtype = torch.float
        bounds = torch.tensor([[-3.0] * d, [3.0] * d], device=gp_device, dtype=dtype)
        BO_BATCH_SIZE = args.n_queries
        N_STEPS = args.n_steps
        MC_SAMPLES = 2000
        
        
        torch.manual_seed(seed)
        best_observed = []
        state_dict = None
        
        # Generate initial data 
        df = pd.read_csv(os.path.join(script_dir,'..','data',args.init_samples)) 
        df=df.sample(args.n_init, random_state = seed) # n_init Initial samples 
        smiles = [s for s in df.smiles if Chem.MolFromSmiles(s)!=None]
        
        loader.graph_only=True
        train_z = torch.tensor(model.embed( loader, df)) # z has shape (N_molecules, latent_size)
        
        if args.objective == 'cqed':
            scores_init = torch.tensor([cQED(s) for s in smiles]).view(-1,1).to(device)
        elif args.objective == 'clogp':
            scores_init = torch.tensor([cLogP(s) for s in smiles]).view(-1,1).to(device)
        elif args.objective == 'aff_pred':
            with torch.no_grad():
                scores_init = -1* model.affs(train_z.to(device)).cpu() # careful, maximize -aff <=> minimize binding energy (negative value)
        elif args.objective == 'aff' : 
            PYTHONSH, VINA = set_path(args.server)
            scores_init = -1* torch.tensor(df.drd3).view(-1,1).cpu() # careful, maximize -aff <=> minimize binding energy (negative value)
        elif args.objective == 'qsar':
            with open(os.path.join(script_dir, '..', args.qsar), 'rb') as f :
                clf = pickle.load(f)
                print('-> Loaded qsar svm')
                fps=[]
                for s in smiles:
                    m = Chem.MolFromSmiles(s)
                    fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048)).reshape(1,-1))
                fps = np.vstack(fps)
                scores_init = torch.from_numpy(clf.predict_proba(fps)[:,1]).view(-1,1)
                
        
        # Tracing results logging
        sc_dict = {}
        best_value = torch.max(scores_init).item()
        best_observed.append(best_value)
        train_obj = scores_init
        train_smiles = list(smiles)
        print(f'-> Best value observed in initial samples : {best_value}')
        
        out_dir = os.path.join(script_dir,'results',args.bo_name, f'simulation_{seed}')
        soft_mkdir(os.path.join(script_dir,'results', args.bo_name))
        soft_mkdir(out_dir) 
        
        save_csv = os.path.join(out_dir, 'samples.csv') # csv to write samples and their score 
        header = ['smiles',args.objective, 'step']
        with open(save_csv, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow(header)
        
        with open(save_csv, 'a', newline='') as csvfile:
            for i,s in enumerate(smiles):
                csv.writer(csvfile).writerow([s, scores_init[i,0].item(), 0])
    
        
        # Acquisition function 
        def dock_one(enum_tuple):
            """ Docks one smiles. Input = tuple from enumerate iterator"""
            identifier, smiles = enum_tuple
            return dock(smiles, identifier, PYTHONSH, VINA, parallel=False, exhaustiveness = args.ex)
        
        
        def optimize_acqf_and_get_observation(acq_func, device, gp_device):
            """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
            
            # optimize
            queries_start_time = time()
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=torch.stack([
                    torch.zeros(d, dtype=dtype, device=gp_device), 
                    torch.ones(d, dtype=dtype, device=gp_device),
                ]),
                q=BO_BATCH_SIZE,
                num_restarts=10,
                raw_samples=200,
            )
            queries_stop_time = time()
            
            print('TIME TO DRAW SAMPLES : ', queries_stop_time-queries_start_time)
        
            # observe new values 
            new_z = unnormalize(candidates.detach(), bounds=bounds).to(device)
            
            # Decode z into smiles
            with torch.no_grad():
                gen_seq = model.decode(new_z)
                smiles = model.probas_to_smiles(gen_seq)
            if vocab=='selfies' :
                k = []
                for s in smiles :
                    s = decoder(s)
                    m = Chem.MolFromSmiles(s)
                    try:
                        Chem.Kekulize(m)
                        s= Chem.MolToSmiles(m, kekuleSmiles = True)
                    except:
                        pass
                    k.append(s)
                smiles = k 
                
                
            if BO_BATCH_SIZE > 1 : # query a batch of smiles 
                
                if args.objective == 'aff':
                    
                    # Multiprocessing
                    pool = Pool()
                    done = np.array([bool(s in load_dict) for s in smiles])
                    done, todo = np.where(done ==True)[0], np.where(done==False)[0]
                    
                    done_smiles = [smiles[i] for i in done]
                    todo_smiles = [smiles[i] for i in todo]
                    
                    new_docking = pool.map(dock_one, enumerate(todo_smiles))
                    pool.close()
                    
                    # Get scores in the right order 
                    new_scores = torch.zeros((BO_BATCH_SIZE), dtype=torch.float)
                    for i, i_glob in enumerate(todo):
                        new_scores[i_glob]=new_docking[i]
                    for i, i_glob in enumerate(done) :
                        new_scores[i_glob]=load_dict[done_smiles[i]] # lookup corresponding smiles in dict 
                        
                    
                    # Update dict with scores 
                    for i,s in enumerate(todo_smiles):
                        load_dict[todo_smiles[i]]= new_docking[i]
                    
                    new_scores = -1* new_scores.unsqueeze(-1) # new scores must be (N*1)
                    
                    
                elif args.objective == 'aff_pred':
                    with torch.no_grad():
                        new_scores = -1* model.affs(new_z).cpu() # shape N*1
                        for i in range(len(smiles)):
                            if not isValid(smiles[i]):
                                new_scores[i,0]=0.0 # assigning 0 score to invalid molecules for realistic oracle 
                    
                elif args.objective =='cqed':
                    
                    new_scores = [cQED(s) for s in smiles]
                    new_scores = torch.tensor(new_scores, dtype = torch.float)
                    new_scores = new_scores.unsqueeze(-1)  # new scores must be (N*1)
                    
                elif args.objective == 'clogp':
                    
                    new_scores = [cLogP(s) for s in smiles]
                    new_scores = torch.tensor(new_scores, dtype = torch.float)
                    new_scores = new_scores.unsqueeze(-1)  # new scores must be (N*1)
                    
                elif args.objective == 'qsar':
                    
                    fps=[]
                    bads = []
                    for i, s in enumerate(smiles):
                        m = Chem.MolFromSmiles(s)
                        if m==None:
                            bads.append(i)
                            fps.append(np.zeros((1,2048), dtype=np.float32))
                        else:
                            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048)).reshape(1,-1))
                            
                    fps = np.vstack(fps)
                    new_scores = torch.from_numpy(clf.predict_proba(fps)[:,1]).view(-1,1) # new scores must be (N*1) 
                    for i in bads :
                        new_scores[i,0]=0.0
                    
                return smiles, new_z, new_scores
            
            else: #query an individual smiles 
                raise NotImplementedError
        
        # ========================================================================
        # run N_BATCH rounds of BayesOpt after the initial random batch
        # ========================================================================
        print('-> Invalid molecules get score 0.0')
        samples_score = [] # avg score of new samples at each step 
        
        for iteration in range(1,N_STEPS+1):    
            print(f'Iter [{iteration}/{N_STEPS}]')
            
            #debug_memory()
            
            # fit the model
            train_start = time()
            train_z =train_z.to(gp_device)
            train_obj=train_obj.to(gp_device)
            
            GP_model = get_fitted_model(
                normalize(train_z, bounds=bounds), 
                standardize(train_obj), 
                state_dict=state_dict,
            )
            
            train_end = time()
            print('TIME TO TRAIN GP: ', train_end - train_start)
            
            # define the qNEI acquisition module using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=seed)
            qEI = qExpectedImprovement(model=GP_model, sampler=qmc_sampler, best_f=standardize(train_obj).max())
        
            # optimize and get new observation
            new_smiles, new_z, new_score = optimize_acqf_and_get_observation(qEI, device, gp_device)
            
            # Save epoch samples 
            with open(save_csv, 'a', newline='') as csvfile:
                for i,s in enumerate(new_smiles):
                    csv.writer(csvfile).writerow([s, new_score[i,0].item(), iteration])
            
            # save acquired scores for next time 
            if args.verbose:
                print(' oracle outputs:')
                print(new_score.numpy())
            sc_dict[iteration]=new_score.numpy()
        
            # update training points
            
            train_smiles+= new_smiles 
            train_z = torch.cat((train_z, new_z.to(gp_device)), dim=0)
            train_obj = torch.cat((train_obj, new_score.to(gp_device)), dim=0)
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
            
            # Save updated dict with docking scores 
            if args.objective =='aff':
                with open(os.path.join(script_dir,'..','docking', args.load), 'wb') as f:
                    pickle.dump(load_dict,f)
        
        
    

    
    
        
        
