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

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

from docking.docking import dock, set_path


def one_slurm(list_smiles, dump_path, pythonsh, vina, unique_id, parallel=True, exhaustiveness=16, mean=False,
              load=False):
    """

    :param dump_path:
    :param parallel:
    :param exhaustiveness:
    :param mean:
    :return:
    """
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


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", default='mac', help="Server to run the docking on, for path and configs.")
    parser.add_argument("-e", "--ex", default=64, help="exhaustiveness parameter for vina")
    args, _ = parser.parse_known_args()

    PYTHONSH, VINA = set_path(args.server)

    # ========== SLURM =============
    # proc_id, num_procs = int(sys.argv[1]), int(sys.argv[2])
    proc_id, num_procs = 2, 3

    dirname = os.path.join(script_dir, 'results','docking_small_results')
    if not os.path.isdir(dirname):
        try:
            os.mkdir(dirname)
        except:
            pass

    # Memoization
    whole_path = os.path.join(script_dir, '..', '..', 'data', 'whole_docking_memo.p')
    docking_whole_results = pickle.load(open(whole_path, 'rb'))

    # parse the docking task of the whole job array and split it
    dump_path = os.path.join(script_dir, 'results/samples.p')
    list_smiles, _ = pickle.load(open(dump_path, 'rb'))

    N = len(list_smiles)
    chunk_size = N // (num_procs - 1)
    chunk_min, chunk_max = proc_id * chunk_size, min((proc_id + 1) * chunk_size, N)
    list_data = list_smiles[chunk_min:chunk_max]

    # Do the docking and dump results
    one_slurm(list_data,
              pythonsh=PYTHONSH,
              vina=VINA,
              dump_path=os.path.join(dirname, f"{proc_id}.csv"),
              unique_id=proc_id,
              parallel=True,
              exhaustiveness=args.ex,
              mean=True,
              load=docking_whole_results)
