# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 10:48:10 2020

@author: jacqu

Add docking scores column to moses_train dataset.
Molecules with no score get 0.0 score. All other scores are negative (interact° energy)
"""

import pandas as pd
import pickle
import argparse
import os
import numpy as np

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.realpath(__file__))

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--binning', help="Bin docking scores as 0 (no data or average score, no loss), 1 (bad) or 2 (good)", 
                        type=bool, default=True)
    
    # ================================
    args=parser.parse_args()
    """

    with open(os.path.join(script_dir, '../data/drd3_scores.pickle'), 'rb') as f:
        scored_smiles = pickle.load(f)

    df = pd.read_csv(os.path.join(script_dir, 'moses_train.csv'), index_col=0)

    smiles = df.smiles

    scores = []
    scores_raw = []
    low = -9
    high = -7

    print('>>> Adding scores to moses_train dataset')
    for s in smiles:
        if s in scored_smiles:

            scores_raw.append(scored_smiles[s])
            if scored_smiles[s] < low:  # active
                scores.append(2)
            elif scored_smiles[s] > high:  # inactive
                scores.append(1)
            else:
                scores.append(0)
        else:
            scores.append(0)
            scores_raw.append(0)

    df['drd3'] = pd.Series(scores_raw, index=df.index)
    df['drd3_binned'] = pd.Series(scores, index=df.index)

    df.to_csv(os.path.join(script_dir, '../data/moses_train.csv'))
    print('Saved train set csv with "drd3" and "drd3_binned" columns. Value 0 indicates no data or uninformative score')

    cpt = np.count_nonzero(df, axis=0)
    print('Number of non zero raw docking scores :', cpt[-2])
    print('Number of non zero binned scores (excluding uninformative scores): ', cpt[-1])

    """
    # test set (no docking scores for now)
    df = pd.read_csv(os.path.join(script_dir,'moses_test.csv'), index_col= 0)
    smiles = df.smiles
    scores = []
    scores_raw = []
    
    print('>>> Adding scores to moses_test dataset')
    for s in smiles :
        if s in scored_smiles : 
            
            scores_raw.append(scored_smiles[s])
            if scored_smiles[s] < low: #active 
                scores.append(2)
            elif scored_smiles[s] >high : #inactive
                scores.append(1)
            else:
                scores.append(0)
        else:
            scores.append(0)
            scores_raw.append(0)
            
    df['drd3'] = pd.Series(scores_raw, index = df.index)
    df['drd3_binned'] = pd.Series(scores, index = df.index)
    
    df.to_csv(os.path.join(script_dir,'moses_test.csv'))
    print('Saved test subset with "drd3" column. Value 0.0 indicates no data')
    """
