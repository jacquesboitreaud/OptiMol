# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:33:44 2020

@author: jacqu

Same as get selfies but with chunks. Annotates a bunch of csv in a dir with their selfies
    and build the alphabet over the whole thing

"""
import pandas as pd
import argparse
from multiprocessing import Pool
import os
import sys
from ordered_set import OrderedSet
from tqdm import tqdm
import json

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..'))

from data_processing.get_selfies import process_one


def add_selfies(dir_path='data/fabritiis/input_data/', alphabet='fabritiis.json', serial=False, save_path=None):
    largest_smiles_len = 0
    largest_selfies_len = 0

    smiles_alphabet = OrderedSet()
    selfies_alphabet = OrderedSet()

    if save_path is None:
        save_path = dir_path

    for file in tqdm(os.listdir(dir_path)):
        load_path = os.path.join(dir_path, file)
        df = pd.read_csv(load_path)
        smiles = df['smiles']

        if serial:
            smiles_list = []
            selfies_list = []
            smiles_lengths = []
            selfies_lengths = []
            for s in tqdm(smiles):
                smile, selfie, smile_len, selfie_len = process_one(s)
                smiles_list.append(smile)
                selfies_list.append(selfie)
                smiles_lengths.append(smile_len)
                selfies_lengths.append(selfie_len)

        else:
            pool = Pool()
            res_lists = pool.map(process_one, smiles)
            smiles_list, selfies_list, smiles_lengths, selfies_lengths = map(list, zip(*res_lists))

        # print(time.perf_counter()-time1)

        df = pd.DataFrame.from_dict({'smiles': smiles_list, 'selfies': selfies_list})
        savepath = os.path.join(save_path, file)
        df.to_csv(savepath)

        largest_smiles_len = max(max(smiles_lengths), largest_smiles_len)
        largest_selfies_len = max(max(selfies_lengths), largest_selfies_len)

        # Alphabets and longest smiles / selfies collection

        # This is character based parsing
        local_smiles_alphabet = OrderedSet(''.join(smiles_list))

        # This is selfies parsing
        selfies_alphabet_pre = OrderedSet(''.join(selfies_list)[1:-1].split(']['))
        local_selfies_alphabet = []
        for selfies_element in selfies_alphabet_pre:
            local_selfies_alphabet.append('[' + selfies_element + ']')

        # Merge local alphabets
        smiles_alphabet |= local_smiles_alphabet
        selfies_alphabet |= local_selfies_alphabet

    smiles_alphabet = list(smiles_alphabet)
    selfies_alphabet = list(selfies_alphabet)

    d = {'selfies_alphabet': selfies_alphabet,
         'largest_selfies_len': largest_selfies_len,
         'smiles_alphabet': smiles_alphabet,
         'largest_smiles_len': largest_smiles_len}

    with open(os.path.join(script_dir, '..', 'map_files', alphabet), 'w') as outfile:
        json.dump(d, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dir_path', help="path to csv with dataset", type=str,
                        default='../data/fabritiis/input_data/')
    parser.add_argument('--alphabet', help="Name for alphabet json file saved in map_files", type=str,
                        default='fabritiis.json')

    # ======================
    args, _ = parser.parse_known_args()
    add_selfies(dir_path=args.dir_path, alphabet=args.alphabet, serial=False)
