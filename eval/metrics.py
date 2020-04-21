# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:30:12 2020

@author: jacqu

Moses metrics to evaluate quality and diversity of samples 

"""
import moses 
import os 
import sys
import argparse

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', "-- generated_samples", help="Nbr to generate", type=str, default='data/gen.txt')

    args = parser.parse_args()
    # =======================================
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, 'dataloaders'))
    sys.path.append(os.path.join(script_dir, 'data_processing'))

    with open(os.path.join(script_dir,'..',args.generated_samples), 'r') as f :
        smiles_list = [line.rstrip() for line in f]
        
    print(f'> Read {len(smiles_list)} smiles in data/gen.txt. Computing metrics...')
    metrics = moses.get_all_metrics(smiles_list)
    
    print('MOSES benchmark metrics :')
    for k,v in metrics.items():
        print(k,':', f'{v:.4f}')
    
    # to copy values to excel sheet with benchmarks 
    for k,v in metrics.items():
        print( f'{v:.4f}')
