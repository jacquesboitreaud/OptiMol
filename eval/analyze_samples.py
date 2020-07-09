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

from eval.eval_utils import plot_csvs
from utils import soft_mkdir
from data_processing.comp_metrics import cycle_score, logP, qed
from data_processing.sascorer import calculateScore
from multiprocessing import Pool


parser = argparse.ArgumentParser()
parser.add_argument('--csv_name', type=str, default='data/diverse_samples.csv') # model name in results/saved_models/
# =======

args, _ = parser.parse_known_args()

samples = pd.read_csv(os.path.join(script_dir,'..', args.csv_name))
    
percentage_cutoff = 0.1 # compute intdiv on top 10% molecules in the samples 
N = int(samples.shape[0]*percentage_cutoff)

smiles = samples.smile
smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
mols = [Chem.MolFromSmiles(s) for s in smiles]

### Get QED distribution at different steps of CbAS
print('>>> QED distrib')

def process_one(m):
    q = QED.default(m)
    return q

pool = Pool()
q = pool.map(process_one, mols)
sns.distplot(q)

### Tanimoto pairwise distances 
print('>>> diversity')

fps = [AllChem.GetMorganFingerprintAsBitVect(m , 3, nBits=2048) for m in mols]
fps= np.array(fps)
D= pairwise_distances(fps, metric = 'jaccard')
D_top = D[:N,:N]

print('Tanim pairwise dist overall : ', np.mean(D))
print('Tanim pairwise dist in top:', np.mean(D_top))

### Docking scores 
try : 
    scores = samples.score
except : 
    print('Dataframe has no column "score", ignoring the docking scores analysis')
    sys.exit()
    
sns.distplot(scores)

samples = samples.sort_values('score')
smiles, scores = samples.smile, samples.score

smiles = smiles[:50]
scores = scores[:50]
mols = [Chem.MolFromSmiles(s) for s in smiles]

img = Draw.MolsToGridImage(mols, molsPerRow= 5, legends = [f'{sc:.2f}' for sc in scores])


    

