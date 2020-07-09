# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:44:07 2020

@author: jacqu

Script to plot best generated compounds, and compute enrichments wrt. the prior distribution (here Moses 130k scored compounds)
"""

from rdkit import Chem
from rdkit.Chem import Draw
import pickle
import pandas as pd
import numpy as np


def draw_multi(tuples):
    # list of smiles 
    mols=[Chem.MolFromSmiles(t[0]) for t in tuples]
    mols = [m for m in mols if m!=None]
    img = Draw.MolsToGridImage(mols, molsPerRow=7, maxMols=75, subImgSize=(100, 100), legends=[f'{t[1]:.2f}' for t in tuples] )
    return img

## Load data 

with open('dic_jacques.p', 'rb') as f:
    d = pickle.load(f)
    
df = pd.read_csv('../../data/moses_scored.csv')
df=df.sort_values('drd3')

### Fraction < -10 : 

scores = df.drd3
goods = np.where(scores<=-10)
print('frac <-10 in random : ', goods[0].shape[0]/df.shape[0])

mols = sorted(d.items(), key=lambda item: item[1])

d= mols
good_generated = [t[1] for t in d if t[1]<=-10.0]
print('frac <-10 in gen: ', len(good_generated)/len(d))

### Plots of top N mols : 
N=50

mols=[Chem.MolFromSmiles(m[0]) for m in mols]
mols = [m for m in mols if m!=None]
mols=mols[:N]
img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(200, 130), legends=[f'{t[1]:.2f}' for t in d])

#img.save('mols.png')

### Top N from random sample
mols_moses = df.iloc[:N]
mols_moses, moses_sc = mols_moses.smiles, mols_moses.drd3
mols_moses = [Chem.MolFromSmiles(s) for s in mols_moses]
img = Draw.MolsToGridImage(mols_moses, molsPerRow=5, subImgSize=(200, 130), legends=[f'{s:.2f}' for s in moses_sc])

### Enrichment : 

subsamp = df.sample(100000)
scores = subsamp.drd3
tup = [(k,0) for k in scores]

gen_tup = [(v[1],1) for v in d]

merged = tup+gen_tup
frac_init = len(gen_tup)/(len(tup)+len(gen_tup))
print('Frac init of generated compounds in the sample :', frac_init)
merged = sorted(merged, key = lambda t : t[0])

## thresholds : 

for th in [0.001, 0.005, 0.01]:
    n = int(th*len(merged))
    tops = merged[:n]
    
    frac_gen = np.sum(np.array([t[1] for t in tops]))/n
    
    print(f'> Frac generated at threshold {th}: {frac_gen}')
    print('EF = ', frac_gen/frac_init)
    

