# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Evaluate training of Graph2Smiles VAE (RGCN encoder, GRU decoder, beam search decoding). 

Open multitask affinity model and test it 

Plot all diagnostic plots 


"""
import os
import sys
import argparse

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import torch

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import DataStructs
from selfies import decoder

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances, silhouette_score

from joblib import dump, load
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Name of saved model directory, in /results/saved_models",
                    default='search_vae')
parser.add_argument('-i', '--test_set', help="Test molecules file, in /data",
                    default='moses_test.csv')
parser.add_argument('-N', '--n_mols', help="Number of molecules, set to -1 for all in csv ", type = int, 
                    default=1000)

args = parser.parse_args()
if __name__ == "__main__":
    from dataloaders.molDataset import Loader
    from model import Model, model_from_json
    from loss_func import VAELoss

    from eval.eval_utils import *
    from utils import *
    from dgl_utils import * 

    # Should be same as for training
    properties = ['QED', 'logP', 'molWt']
    targets = ['drd3']

    # Select only DUDE subset to plot in PCA space 
    plot_target = 'drd3'

    # Load eval set
    loaders = Loader(csv_path=os.path.join(script_dir, 'data', args.test_set),
                     maps_path= os.path.join(script_dir, 'map_files'),
                     n_mols=args.n_mols,
                     vocab='selfies',
                     num_workers=0,
                     batch_size=100,
                     props=properties,
                     targets=targets,
                     test_only=True)
    rem, ram, rchim, rcham = loaders.get_reverse_maps()
    _, _, test_loader = loaders.get_data()

    # Model loading 
    model = model_from_json(args.name)
    device = model.device
    model.eval()

    # Smiles 
    out_all = np.zeros((loaders.dataset.n, loaders.dataset.max_len))
    smi_all = np.zeros((loaders.dataset.n, loaders.dataset.max_len))

    # Latent embeddings
    z_all = np.zeros((loaders.dataset.n, model.l_size))

    # Affinities
    a_target_all = np.zeros((loaders.dataset.n, len(targets)))
    a_all = np.zeros((loaders.dataset.n, len(targets)))

    # Mol Props
    p_target_all = np.zeros((loaders.dataset.n, len(properties)))
    p_all = np.zeros((loaders.dataset.n, len(properties)))

    with torch.no_grad():
        for batch_idx, (graph, smiles, p_target, a_target) in enumerate(test_loader):
            graph = send_graph_to_device(graph, device)
            smiles=smiles.to(device)

            # Latent embeddings
            mu, logv, z, out, pred_p, pred_a = model(graph, smiles, tf=0 )

            # Decoding 
            v, indices = torch.max(out, dim=1)  # get indices of char with max probability
            
            # Evaluate loss 
            rec, kl = VAELoss(out, indices, mu, logv)
            print('Rec batch loss : ', rec)
            print('KL batch loss : ', kl)

            # Decoding with beam search 
            """
            beam_output = model.decode_beam(z,k=3, cutoff_mols=10)
            # Only valid out molecules 
            mols, _ = log_from_beam(beam_output,loaders.dataset.index_to_char)
            """

            # Concat all input and ouput data 
            batch_size = z.shape[0]
            indices = indices.cpu().numpy()
            out_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = indices
            smi_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = smiles

            z = z.cpu().numpy()
            z_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = z

            props = pred_p.cpu().numpy()
            p_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = props

            p_target_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = p_target.numpy()

            # Classification of binned affinities
            affs = pred_a.cpu().numpy()
            a_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = affs

            a_target_all[batch_idx * batch_size:(batch_idx + 1) * batch_size] = a_target.numpy()

        # ===================================================================
        # Latent space KDE 
        # ===================================================================

        plot_kde(z_all)

        D_a = pairwise_distances(z_all, metric='cosine')
        print('Average cosine distance in latent space : ', np.mean(D_a))

        # ===================================================================
        # Decoding statistics 
        # ===================================================================
        reconstruction_dataframe, frac_valid, frac_id = log_smiles_from_indices(smi_all, out_all,
                                                                                loaders.dataset.index_to_char)
        # only for smiles
        # print("Fraction of molecules decoded into a valid smiles: ", frac_valid)
        # print("Fraction of perfectly reconstructed mols ", frac_id)
        smiles = [decoder(s) for s in reconstruction_dataframe['output smiles']]

        # ===================================================================
        # Prediction error plots 
        # ===================================================================

        for i, p in enumerate(properties):
            reconstruction_dataframe[p] = p_target_all[:, i]
            reconstruction_dataframe[p + '_pred'] = p_all[:, i]

        for i, t in enumerate(targets):
            reconstruction_dataframe[t] = a_target_all[:, i]
            reconstruction_dataframe[t + '_pred'] = a_all[:, i]
        reconstruction_dataframe = reconstruction_dataframe.replace([np.inf, -np.inf], 0)

        plt.figure()
        sns.scatterplot(reconstruction_dataframe['QED'], reconstruction_dataframe['QED_pred'])
        sns.lineplot([0, 1], [0, 1], color='r')

        plt.figure()
        sns.scatterplot(reconstruction_dataframe['logP'], reconstruction_dataframe['logP_pred'])
        sns.lineplot([-2, 5], [-2, 5], color='r')

        plt.figure()
        sns.scatterplot(reconstruction_dataframe['molWt'], reconstruction_dataframe['molWt_pred'])
        sns.lineplot([0, 700], [0, 700], color='r')

        # Affinities prediction plots
        plt.figure()
        plt.xlim(-12,-5)
        sns.scatterplot(reconstruction_dataframe[targets[0]], reconstruction_dataframe[f'{targets[0]}_pred'])
        sns.lineplot([-12, -5], [-12, -5], color='r')

        MAE = np.mean(np.abs(reconstruction_dataframe[targets[0]] - reconstruction_dataframe[f'{targets[0]}_pred']))
        print(f'MAE = {MAE} kcal/mol')

        # ===================================================================
        # PCA plot 
        # ===================================================================

        try:
            fitted_pca = load( os.path.join(script_dir,'results/saved_models',args.name,'fitted_pca.joblib'))
        except(FileNotFoundError):
            print(
                'Fitted PCA object not found at /data/fitted_pca.joblib, new PCA will be fitted on current data.')
            fitted_pca = fit_pca(z)

        # Plot PCA with desired hue variable 
        plt.figure()
        pca_plot_hue(z=z_all, pca=fitted_pca, variable=p_target_all[:, 1], label = 'logP')
        
        plt.figure()
        pca_plot_hue(z=z_all, pca=fitted_pca, variable=p_target_all[:, 2],  label = 'Weight')
        
        plt.figure()
        pca_plot_hue(z=z_all, pca=fitted_pca, variable=p_target_all[:, 0], label = 'QED')

        # PCA Affinities
        plt.figure()
        ax = pca_plot_hue(z=z_all, pca=fitted_pca, variable=a_all[:, 0],  label = 'Predicted docking')
        left, right = ax.get_xlim()
        down,up = ax.get_ylim()

        # ====================================================================
        # Random sampling in latent space 
        # ====================================================================
        
        Nsamples = 1000
        r = torch.tensor(np.random.normal(size=(Nsamples,model.l_size)), dtype=torch.float)        
        # PCA plot 
        plt.figure()
        plt.xlim(left, right)
        plt.ylim(down, up)
        pca_plot_color(z=r,  pca=fitted_pca, color = 'red', label = 'random normal')
        plt.title('Random normal samples in PCA space')
        
        
        # Decode 
        out = model.decode(r)
        selfies = model.probas_to_smiles(out)
        s = [decoder(se) for se in selfies]
        
        # Unique smiles 
        u = np.unique(s)
        N_uniques = u.shape[0]
        print(f'Number unique smiles in {Nsamples}: {N_uniques}')
        

        mols = [Chem.MolFromSmiles(smi) for smi in u]
        mols = [m for m in mols if m!=None]
        qed = [Chem.QED.qed(m) for m in mols]
        fps = [Chem.RDKFingerprint(x) for x in mols]
        
        
        fig = Draw.MolsToGridImage(mols[:100], legends = [f'{q:.2f}' for q in qed])
        """
        for m in mols:
            fig = Draw.MolToMPL(m, size = (100,100))
        """
        
        i1 = 1
        
        cpt=0.
        cpt_similar = 0 
        for i in range(len(fps)) :
            sim = DataStructs.FingerprintSimilarity(fps[i1],fps[i], metric = DataStructs.TanimotoSimilarity)
            cpt+=sim
            if(sim >0.8):
                cpt_similar +=1 
        print(f'{cpt_similar} molecules with Tanimoto sim > 0.8 to mol n°{i1}')
        print(f'{cpt/len(mols)} average similarity to seed molecule')
        
        
