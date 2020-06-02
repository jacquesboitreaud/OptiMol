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
from rdkit.Chem import AllChem
import os
import sys
from ordered_set import OrderedSet

import json

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.neutralize import NeutraliseCharges
from selfies import encoder, decoder


def clean_smiles(s):
    """ Function to clean smiles; change as needed """
    s2 = NeutraliseCharges(s)
    m = AllChem.MolFromSmiles(s2[0])
    Chem.Kekulize(m)
    s = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=True)
    return s


def process_one(s):
    clean_smile = clean_smiles(s)

    individual_selfie = encoder(clean_smile)
    s_len = len(individual_selfie) - len(individual_selfie.replace('[', ''))
    return clean_smile, individual_selfie, len(clean_smile), s_len


def add_selfies(path='data/moses_train.csv', alphabet='my_alphabet.json', serial=False, smi=False):
    if not smi:
        train = pd.read_csv(path, index_col=0)
        smiles = train.smiles
    else:
        with open(path, 'r') as f:
            smiles = f.readlines()
            smiles = [s.rstrip() for s in smiles]

    # time1 = time.perf_counter()

    if serial:
        smiles_list = []
        selfies_list = []
        max_smiles_len = []
        max_selfies_len = []
        for s in tqdm(smiles):
            smile, selfie, smile_len, selfie_len = process_one(s)
            smiles_list.append(smile)
            selfies_list.append(selfie)
            max_smiles_len.append(smile_len)
            max_selfies_len.append(selfie_len)

    else:
        pool = Pool()
        res_lists = pool.map(process_one, smiles)
        smiles_list, selfies_list, max_smiles_len, max_selfies_len = map(list, zip(*res_lists))

    # print(time.perf_counter()-time1)

    if not smi:
        train['selfies'] = pd.Series(selfies_list, index=train.index)
        train['smiles'] = pd.Series(smiles_list, index=train.index)
        train.to_csv(path)

    else:
        train = pd.DataFrame.from_dict({'smiles': smiles_list, 'selfies': selfies_list})
        print(train)
        train.to_csv(path[:-4] + '.csv')

    # Alphabets and longest smiles / selfies collection 

    smiles_alphabet = list(OrderedSet(''.join(smiles_list)))
    largest_smiles_len = len(max(smiles_list, key=len))
    selfies_len = []

    print(f'--> Building alphabets for {smiles_list.shape[0]} smiles and selfies in dataset...')
    for individual_selfie in selfies_list:
        selfies_len.append(len(individual_selfie) - len(individual_selfie.replace('[', '')))  # len of SELFIES
    selfies_alphabet_pre = list(OrderedSet(''.join(selfies_list)[1:-1].split('][')))
    selfies_alphabet = []
    for selfies_element in selfies_alphabet_pre:
        selfies_alphabet.append('[' + selfies_element + ']')
    largest_selfies_len = max(selfies_len)

    print('Finished parsing smiles and selfies alphabet. Saving to json file custom_alphabets.json')
    print('Longest selfies : ', largest_selfies_len)
    print('Longest smiles : ', largest_smiles_len)

    d = {'selfies_alphabet': selfies_alphabet,
         'largest_selfies_len': largest_selfies_len,
         'smiles_alphabet': smiles_alphabet,
         'largest_smiles_len': largest_smiles_len}

    with open(os.path.join(script_dir, '..', 'map_files', alphabet), 'w') as outfile:
        json.dump(d, outfile)
    print('Saved alphabet to ', alphabet)

    print('Saved selfies as a column in csv. Ready to train with selfies.')
    print(f'Max smiles length : {max(max_smiles_len)}')
    print(f'Max selfies length : {max(max_selfies_len)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--smiles_file', help="path to csv with dataset", type=str,
                        default='../data/250k_zinc.smi')

    parser.add_argument('--smi', help="input smi file", action='store_true',
                        default=True)
    parser.add_argument('--alphabet', help="Name for alphabet json file saved in map_files", type=str,
                        default='my_alphabets.json')

    # ======================
    args, _ = parser.parse_known_args()

    # A sanity check for the 'clean_smiles' function in use : 
    print('Showing 3 sample smiles to check stereo and charges handling :')
    smiles = ['CC(=O)C1=CC=CC=C1CNCCS1C=NC=N1', 'C=CCN1C(=O)/C(=C/c2ccccc2F)S/C1=N\S(=O)(=O)c1cccs1',
              'N#Cc1ccnc(N2CCC([NH2+]C[C@@H]3CCCO3)CC2)c1']
    for s in smiles:
        s = clean_smiles(s)
        print(s)

    print(f'>>> Computing selfies for all smiles in {args.smiles_file}. May take some time.')

    if args.smi:
        print('Taking .smi file as input, output to a csv')
    add_selfies(path=args.smiles_file, smi=args.smi, alphabet=args.alphabet, serial=False)
