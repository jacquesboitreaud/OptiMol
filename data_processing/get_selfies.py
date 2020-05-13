# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Compute selfies for all smiles in csv 
"""
import pandas as pd
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from rdkit import Chem
import os, sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

from selfies import encoder, decoder, selfies_alphabet


def process_one(s):
    m = Chem.MolFromSmiles(s)
    c = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    individual_selfie = encoder(c)
    s_len = len(individual_selfie) - len(individual_selfie.replace('[', ''))
    return individual_selfie, len(c), s_len


def process_one_kekule(s):
    m = Chem.MolFromSmiles(s)
    Chem.Kekulize(m)
    s = Chem.MolToSmiles(m, kekuleSmiles=True)
    individual_selfie = encoder(s)
    s_len = len(individual_selfie) - len(individual_selfie.replace('[', ''))
    return individual_selfie, len(s), s_len


def add_selfies(path='data/moses_train.csv', serial=False, kekule=False):
    train = pd.read_csv(path, index_col=0)

    # time1 = time.perf_counter()

    if serial:
        selfies_list = []
        max_smiles_len = []
        max_selfies_len = []
        for s in tqdm(train.smiles):
            if not kekule:
                selfie, smile_len, selfie_len = process_one(s)
            else:
                selfie, smile_len, selfie_len = process_one_kekule(s)
            selfies_list.append(selfie)
            max_smiles_len.append(smile_len)
            max_selfies_len.append(selfie_len)

    else:
        pool = Pool()
        if not kekule:
            res_lists = pool.map(process_one, train.smiles)
        else:
            res_lists = pool.map(process_one_kekule, train.smiles)
        selfies_list, max_smiles_len, max_selfies_len = map(list, zip(*res_lists))

    # print(time.perf_counter()-time1)

    train['selfies'] = pd.Series(selfies_list, index=train.index)
    train.to_csv(path)
    print('Saved selfies as a column in csv. Ready to train with selfies.')
    print(f'Max smiles length : {max(max_smiles_len)}')
    print(f'Max selfies length : {max(max_selfies_len)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--smiles_file', help="path to csv with dataset", type=str,
                        default='data/moses_scored_valid.csv')
    parser.add_argument('--k', help="Kekule smiles", action='store_true',
                        default=True)

    # ======================
    args, _ = parser.parse_known_args()

    print(f'>>> Computing selfies for all smiles in {args.smiles_file}. May take some time.')
    if args.k:
        print('Careful. Converting to kekule smiles before selfies encoding. ')
    add_selfies(args.smiles_file, kekule=args.k)
