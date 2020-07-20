# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:30:23 2020

@author: jacqu

Compute statistics on 100k samples from cbas model 
"""

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import argparse

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Draw
from rdkit import DataStructs
from rdkit.Chem import AllChem
from scipy.spatial.distance import jaccard

from sklearn.metrics import pairwise_distances

from cairosvg import svg2pdf



from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore
from multiprocessing import Pool

if __name__ == "__main__":
    
    optimol = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/big_new_lr/optimol_scored.csv'))
    try:
        multiobj = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/multiobj_big/multiobj_scored.csv'))
    except:
        pass
    gianni =  pd.read_csv(os.path.join(script_dir,'..', 'data/fabritiis_docked.csv'))
    zinc = pd.read_csv(os.path.join(script_dir,'..', 'data/zinc_docked.csv'))
    
    threshold = 0.01
    
    def qeds(df):
        N=df.shape[0]
        n =int(N*threshold)
        df=df.sort_values('score')
        df=df[:n] # top 10 % statistics 
        smiles= df.smile
        smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        q = np.array([QED.default(m) for m in mols])
        
        return (np.mean(q), np.std(q))
    
    def sas(df):
        N=df.shape[0]
        n =int(N*threshold)
        df=df.sort_values('score')
        df=df[:n]
        smiles= df.smile
        smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        q = np.array([calculateScore(m) for m in mols])
        
        return (np.mean(q), np.std(q))
    
    def docking(df): 
        N=df.shape[0]
        n =int(N*threshold)
        df=df.sort_values('score')
        df=df[:n] # top 10 % statistics 
        scores = df.score
        return (np.mean(scores), np.std(scores))
    
    def tanim_dist(df):
        N=df.shape[0]
        n =int(N*threshold)
        df=df.sort_values('score')
        df=df[:n] # top 10 % statistics 
        smiles= df.smile
        smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048) for m in mols]
        fps= np.array(fps)
        D= pairwise_distances(fps, metric = 'jaccard')
        
        return (np.mean(D), np.std(D))
    
    # ZINC : 
    """
    print('*********** ZINC ********')
    
    qed_zinc = qeds(zinc)
    print(qed_zinc)
    sa_zinc = sas(zinc)
    print(sa_zinc)
    docking_zinc = docking(zinc)
    print(docking_zinc)
    
    div_zinc = tanim_dist(zinc)
    print(div_zinc)
    
    # Gianni 
    print('*********** Gianni ********')
    qed_gianni = qeds(gianni)
    print(qed_gianni)
    sa_gianni = sas(gianni)
    print(sa_gianni)
    docking_gianni = docking(gianni)
    print(docking_gianni)
    div_gianni= tanim_dist(gianni)
    print(div_gianni)
    
    # Optimol 
    print('*********** Optimol ********')
    qed_o = qeds(optimol)
    print(qed_o)
    sa_o = sas(optimol)
    print(sa_o)
    docking_o = docking(optimol)
    print(docking_o)
    div_o = tanim_dist(optimol)
    print(div_o)
    
    # Multiobj : TODO 
    print('*********** Multiobj: todo ********')
    qed_m = qeds(multiobj)
    print(qed_m)
    sa_m = sas(multiobj)
    print(sa_m)
    docking_m = docking(multiobj)
    print(docking_m)
    div_m = tanim_dist(multiobj)
    print(div_m)
    
    
    ### Searching for actives : 
    
    actives = pd.read_csv(os.path.join(script_dir,'..','data','excape_drd3.csv'))
    actives = actives[actives['active']=='A']
    actives=actives.smile
    a_mols = [Chem.MolFromSmiles(s) for s in actives]
    a_smiles = [Chem.MolToSmiles(m) for m in a_mols if m is not None]
    a_smiles = set(a_smiles)
    
    cpt = 0
    found = []
    for s in smiles : 
        s= Chem.MolToSmiles(Chem.MolFromSmiles(s)) # canonical
        if s in a_smiles : 
            cpt +=1
            found.append(s)
    print(cpt, ' actives found in samples: ')
    print(found)
    """
    
    N = 3
    
    # Plot : 2500 first : 
    optimol = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/big_new_lr/optimol_scored.csv'))
    optimol = optimol[:10000]
    optimol = optimol.sample(10)
    
    
    # Top molecules 
    samples = optimol.sort_values('score')
    smiles, scores = samples.smile, samples.score
    
    smiles = smiles[:N]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    qeds = np.array([QED.default(m) for m in mols])
    sas = [calculateScore(m) for m in mols]
    scores = scores[:N]
    
    
    img1 = Draw.MolsToGridImage(mols, molsPerRow= 1, useSVG=False, legends = [f'{sc:.2f}' for sc in scores])
    img2 =  Draw.MolsToGridImage(mols, molsPerRow= 1, useSVG=False)
    
    svg2pdf(str(img),write_to='optimol_samp_2.pdf')
    
    """
    ['Cc1ccccc1CCC(=O)OCC(=O)NCCc1ccc2ccccc2c1'
     'COCC1CCCCN(C(=O)C(=O)NCc2cc3ccccc3c3ccccc23)C1'
     'Cc1ccccc1CC1CCCN1C(=O)CC1Cc2ccccc2NC1=O']
    """


