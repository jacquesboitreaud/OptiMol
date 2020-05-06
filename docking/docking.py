import sys
import subprocess
import os
import argparse
from time import time
import numpy as np
import pybel
import pandas as pd
import shutil

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..'))

if __name__=='__main__':
    from utils import soft_mkdir

RECEPTOR_PATH = os.path.join(script_dir, 'data_docking/drd3.pdbqt')
CONF_PATH = os.path.join(script_dir, 'data_docking/conf.txt')


def set_path(computer):
    if computer == 'rup':
        PYTHONSH = '/home/mcb/users/jboitr/local/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
        VINA = '/home/mcb/users/jboitr/local/autodock_vina_1_1_2_linux_x86/bin/vina'
    elif computer == 'cedar':
        PYTHONSH = '/home/jboitr/projects/def-jeromew/docking_setup/mgltools_x86_64Linux2_1.5.6/bin/pythonsh'
        VINA = '/home/jboitr/projects/def-jeromew/docking_setup/autodock_vina_1_1_2_linux_x86/bin/vina'
    elif computer == 'pasteur':
        PYTHONSH = '/c7/home/vmallet/install/mgltools_x86_64Linux2_1.5.7/bin/pythonsh'
        VINA = '/c7/home/vmallet/install/autodock_vina_1_1_2_linux_x86/bin/vina'
    elif computer == 'mac':
        PYTHONSH = '/Users/vincent/bins/mgltools_1.5.7_MacOS-X/bin/pythonsh'
        VINA = '/Users/vincent/bins/vina/bin/vina'
    else:
        print('Error: "server" argument never used before. Set paths of vina/mgltools installs for this server.')
    return PYTHONSH, VINA


def prepare_receptor():
    # just run pythonsh prepare_receptor4.py -r drd3.pdb -o drd3.pdbqt -A hydrogens
    subprocess.run(f"{PYTHONSH} prepare_receptor4.py -r drd3.pdb -o {RECEPTOR_PATH} -A hydrogens".split())


def dock(smile, unique_id, PYTHONSH, VINA, parallel=True, exhaustiveness=16):
    """"""
    soft_mkdir('tmp')
    tmp_path = f'tmp/{str(unique_id)}'
    soft_mkdir(tmp_path)
    try:
        pass
        # PROCESS MOLECULE
        mol = pybel.readstring("smi", smile)
        mol.addh()
        mol.make3D()
        dump_mol2_path = os.path.join(tmp_path, 'ligand.mol2')
        dump_pdbqt_path = os.path.join(tmp_path, 'ligand.pdbqt')
        mol.write('mol2', dump_mol2_path, overwrite=True)
        subprocess.run(f'{PYTHONSH} prepare_ligand4.py -l {dump_mol2_path} -o {dump_pdbqt_path} -A hydrogens'.split())
    
        start = time()
        # DOCK
        if parallel:
            print(f'{VINA} --receptor {RECEPTOR_PATH} --ligand {dump_pdbqt_path}'
                  f' --config {CONF_PATH} --exhaustiveness {exhaustiveness} --log log.txt')
            subprocess.run(f'{VINA} --receptor {RECEPTOR_PATH} --ligand {dump_pdbqt_path}'
                           f' --config {CONF_PATH} --exhaustiveness {exhaustiveness} --log log.txt'.split())
        else:
            subprocess.run(f'{VINA} --config conf.txt --exhaustiveness 16 --log log.txt --cpu 1'.split())
        delta_t = time() - start
        print("Docking time :", delta_t)
    
        with open(os.path.join(tmp_path, 'ligand_out.pdbqt'), 'r') as f:
            lines = f.readlines()
            slines = [l for l in lines if l.startswith('REMARK VINA RESULT')]
            values = [l.split() for l in slines]
            # In each split string, item with index 3 should be the kcal/mol energy.
            score = np.mean([float(v[3]) for v in values])

    except:
        score = 0
    shutil.rmtree(tmp_path)
    print("Score :", score)
    return smile, score


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default='mac', help="Server to run the docking on, for path and configs.")
    parser.add_argument("-e", "--ex", default=16, help="exhaustiveness parameter for vina")
    args = parser.parse_args()

    PYTHONSH, VINA = set_path(args.server)

    dock('CC1C2CCC(C2)C1CN(CCO)C(=O)c1ccc(Cl)cc1', unique_id=2, PYTHONSH= PYTHONSH, VINA=VINA,  exhaustiveness=args.ex )
