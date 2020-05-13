# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:26:06 2019

@author: jacqu

Compute chemical properties for a dataframe
"""

import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import Draw, QED, Crippen, Descriptors, rdMolDescriptors, GraphDescriptors
import pandas as pd
import tqdm
from multiprocessing import Pool


def process_one(s):
    m = Chem.MolFromSmiles(s)
    qed = QED.default(m)
    logP = Crippen.MolLogP(m)
    molWt = Descriptors.MolWt(m)
    return qed, logP, molWt


def add_props(path='data/moses_train.csv', serial=False):
    data = pd.read_csv(path, index_col=0)
    smiles = data.smiles
    d = {}
    prop_names = ['QED', 'logP', 'molWt']
    for name in prop_names:
        d[f'{name}'] = []

    print(f'>>> computing {prop_names} for {len(smiles)} molecules')
    if serial:
        for s in tqdm(smiles):
            m = Chem.MolFromSmiles(s)
            d['QED'].append(QED.default(m))
            d['logP'].append(Crippen.MolLogP(m))
            d['molWt'].append(Descriptors.MolWt(m))
    else:
        pool = Pool()
        res_lists = pool.map(process_one, smiles)
        d['QED'], d['logP'], d['molWt'] = map(list, zip(*res_lists))

    for k in d.keys():
        data[k] = pd.Series(d[k], index=data.index)

    # Drop lines with Nan properties
    data = data.dropna(axis=0, subset=prop_names)

    savename = path
    print(f'>>> saving dataframe to {savename}')
    data.to_csv(savename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help="path to molecules dataset. Column with smiles labeled smiles'",
                        type=str, default='data/moses_train.csv')
    # =======

    args, _ = parser.parse_known_args()

    add_props(args.input)
