# -*- coding: utf-8 -*-
"""
Created on Thu May  7 18:03:42 2020

@author: jacqu

Dock one smiles 
"""

from docking import dock, set_path
import argparse
import os,sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

if __name__ == '__main__':
    from utils import soft_mkdir
    
    RECEPTOR_PATH = os.path.join(script_dir, 'data_docking/drd3.pdbqt')
    CONF_PATH = os.path.join(script_dir, 'data_docking/conf.txt')
    

if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default='O=CC=C(C1C2=NC=C(C=NC3=CC=CC=C3N=C2)C=C1F)NC=C', 
                        help="Smiles to dock")
    parser.add_argument("-s", "--server", default='mac', help="Server to run the docking on, for path and configs.")
    parser.add_argument("-e", "--ex", default=16, help="exhaustiveness parameter for vina")
    
    args, _ = parser.parse_known_args()

    PYTHONSH, VINA = set_path(args.server)
    
    sc = dock(smile=args.input, unique_id=1, pythonsh=PYTHONSH, vina=VINA, parallel=True, exhaustiveness=args.ex)
    print('Score :', sc)

