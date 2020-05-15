"""
Script to be called by a job array slurm task.
Takes the path to a csv and annotate it with docking scores
"""
import os
import sys
import argparse
import pandas as pd
import csv
import pickle
from rdkit import Chem
from rdkit.Chem import QED

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..', '..'))

from docking.docking import dock, set_path


def one_slurm(list_smiles, server, unique_id, parallel=True, exhaustiveness=16, mean=False,
              load=False):
    """

    :param list_smiles:
    :param server:
    :param unique_id:
    :param parallel:
    :param exhaustiveness:
    :param mean:
    :param load:
    :return:
    """
    pythonsh, vina = set_path(server)
    dump_path = os.path.join(dirname, f"{unique_id}.csv")

    header = ['smile', 'score']
    with open(dump_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(header)

    for smile in list_smiles:
        score_smile = dock(smile, unique_id=unique_id, parallel=parallel, exhaustiveness=exhaustiveness, mean=mean,
                           pythonsh=pythonsh, vina=vina, load=load)
        # score_smile = 0
        with open(dump_path, 'a', newline='') as csvfile:
            list_to_write = [smile, score_smile]
            csv.writer(csvfile).writerow(list_to_write)


def one_slurm_qed(list_smiles, unique_id):
    """

    :param list_smiles:
    :param unique_id:
    :return:
    """
    dump_path = os.path.join(dirname, f"{unique_id}.csv")

    header = ['smile', 'score']
    with open(dump_path, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerow(header)

    for smile in list_smiles:
        m = Chem.MolFromSmiles(smile)
        if m is not None:
            score_smile = QED.qed(m)
        else:
            score_smile = 0
        with open(dump_path, 'a', newline='') as csvfile:
            list_to_write = [smile, score_smile]
            csv.writer(csvfile).writerow(list_to_write)


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default='mac', help="Server to run the docking on, for path and configs.")
    parser.add_argument("-e", "--ex", default=64, help="exhaustiveness parameter for vina")
    parser.add_argument('--qed', action='store_true')
    args, _ = parser.parse_known_args()

    try:
        proc_id, num_procs = int(sys.argv[1]), int(sys.argv[2])
    except IndexError:
        print('We are not using the args as usually in docker.py')
        proc_id, num_procs = 2, 10

    # parse the docking task of the whole job array and split it
    dump_path = os.path.join(script_dir, 'results/docker_samples.p')
    list_smiles = pickle.load(open(dump_path, 'rb'))

    N = len(list_smiles)
    chunk_size, rab = N // (num_procs), N % num_procs
    chunk_min, chunk_max = proc_id * chunk_size, min((proc_id + 1) * chunk_size, N)
    list_data = list_smiles[chunk_min:chunk_max]
    # N = chunk_size*num_procs + rab
    # Share rab between procs
    if proc_id < rab:
        list_data.append(list_smiles[-(proc_id + 1)])

    dirname = os.path.join(script_dir, 'results', 'docking_small_results')

    # Just use qed
    if args.qed:
        one_slurm_qed(list_data, proc_id)
    # Do the docking and dump results
    else:
        one_slurm(list_data,
                  server=args.server,
                  unique_id=proc_id,
                  parallel=False,
                  exhaustiveness=args.ex,
                  mean=True)
