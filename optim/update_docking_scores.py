# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:37:28 2020

@author: jacqu

Add collected docking scores to pickle dict 
"""

import pickle
import pandas as pd 
import argparse
import os

script_dir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()


parser.add_argument('--csv', help='csv with new scores to parse', default='results/bo/10_steps_docking/samples.csv')

args, _ = parser.parse_known_args()
# ================================================================

df = pd.read_csv(os.path.join(script_dir, '..', args.csv))

with open('../docking/drd3_scores.pickle', 'rb') as f:
    load_dict = pickle.load(f)

cpt=0
for i, row in df.iterrows():
    s= row['smiles']
    if s not in load_dict :
        load_dict[s]= row['aff']
        cpt+=1
        
print(f' added {cpt} new docking scores. Now {len(load_dict)}')
        
with open('../docking/drd3_scores.pickle', 'wb') as f:
    pickle.dump(load_dict, f)
