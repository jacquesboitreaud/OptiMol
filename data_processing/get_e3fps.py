# -*- coding: utf-8 -*-
"""
Created on Wed May 13 09:56:28 2020

@author: jacqu

Generate e3fp for molecules 
"""

import numpy 
import pandas as pd 
import os, sys
import pickle
import argparse
from e3fp.pipeline import fprints_from_mol, confs_from_smiles, fprints_from_smiles

from glob import glob
from python_utilities.parallel import Parallelizer
from e3fp.conformer.util import smiles_to_dict

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, '..'))

from rdkit import Chem 

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--csv', help="molecules file, in /data",
                    default='moses_test.csv')
parser.add_argument('-N', '--n_mols', help="Number of molecules, set to -1 for all in csv ", type = int, 
                    default=10)
parser.add_argument('--parallel', help="parallelize over processors ", action = 'store_true', 
                    default=True)

args = parser.parse_args()

# ================

if __name__=='__main__':
    
    csv_name = args.csv
    
    # e3fp params 
    fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
    confgen_params = {'max_energy_diff': 20.0, 'first': 3}
    
    df = pd.read_csv(os.path.join(script_dir,'..', 'data', csv_name), nrows = args.n_mols)
    smiles = df.smiles 
    print(f'>>> computing e3fp for {smiles.shape[0]} molecules')
    
    if not args.parallel:
        fprints_list=[]
        for s in smiles[:10] :
            fprints_list.append(fprints_from_smiles(s, "m1", confgen_params=confgen_params))
        
    else:
        # Parallel fingerprinting 
        smiles_iter = ( (s,smiles_id) for smiles_id,s in enumerate(smiles) )
        
        kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
        parallelizer = Parallelizer(parallel_mode="processes")
        fprints_list = parallelizer.run(fprints_from_smiles, smiles_iter, kwargs=kwargs) 
        
    fprints_dict = {t[1][0]:t[0] for t in fprints_list}
    
    # fprints list has len N_mols 
    # for each mol, tuple of 2 elements is saved 
    # t[0] = list of conformer fingerprints
    # t[1] = (smiles, unique_id)
    
    print(fprints_dict)
    with open(os.path.join(script_dir, '..', 'data', 'e3fp_dict.pickle') , 'wb') as f:
        pickle.dump(fprints_dict,f)
    print('Saved e3fps to e3fp_dict.pickle')


    
    
    