# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:24:31 2020

@author: jacqu

Filter generated batch : 
    - keeps only valid smiles
    - keeps only novel molecules 

"""
import os
import sys
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

import argparse
import sys
import torch
import numpy as np

import pickle
import pandas as pd
import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F

if __name__ == "__main__":

    from dataloaders.molDataset import molDataset
    from model import Model
    from utils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', help=".txt file with smiles", type=str,
                        default='gen_batch.txt')
    parser.add_argument('-o', '--output', help=".txt file with smiles", type=str,
                        default='gen_batch_unique.txt')
    parser.add_argument('-df', '--dataframe', action = 'store_true', help='Optional arg to write a csv dataframe')
    

    args = parser.parse_args()

    # ==============
    
    # Load set of non-novel molecules 
    train = pd.read_csv('../data/moses_train.csv')    
    finetune = pd.read_csv('../data/exp/gpcr/gpcr.csv')
    finetune = finetune[finetune['fold'] == 1]

    seen = set(train['can'])
    seen.update(finetune['can'])
    
    novel=[]

    with open(args.input, 'r') as f:
        line=f.readline()
        while line:
            line = f.readline().rstrip()
            # mol object (validity)
            m = Chem.MolFromSmiles(line)
            if(m!=None and line not in seen):
                #print(line)
                novel.append(line)
               
    if(args.dataframe):
        d={'can':novel}
        df=pd.DataFrame.from_dict(d)
        df.to_csv(args.output)
    else:
        with open(args.output, 'w') as f:
             for s in novel:
                f.write(s)
                f.write('\n')
    print(f'wrote output smiles to {args.output}')
        
                
