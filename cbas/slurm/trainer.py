import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse
import pickle
import pandas as pd
import shutil

from cbas.gen_train import GenTrain
from cbas.oracles import deterministic_one
from model import model_from_json
from utils import *


def gather_scores(iteration):
    """
    Gather docking results
    :return:
    """
    dirname = os.path.join(script_dir, 'results', 'docking_small_results')
    dfs = [pd.read_csv(os.path.join(dirname, csv_file)) for csv_file in os.listdir(dirname)]
    merged = pd.concat(dfs)
    dump_path = os.path.join(script_dir, 'results', 'docking_results', f'{iteration}.csv')
    merged.to_csv(dump_path)

    def empty_folder(folder_path):
        folder_path = '/path/to/folder'
        for file_object in os.listdir(folder_path):
            file_object_path = os.path.join(folder_path, file_object)
            if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
                os.unlink(file_object_path)
            else:
                shutil.rmtree(file_object_path)

    empty_folder(dirname)

    molecules = merged['smile']
    scores = merged['score']

    score_dict = dict(zip(molecules, scores))
    return score_dict


def process_samples(score_dict, samples, weights, quantile):
    """
    reweight samples using docking scores
    :return:
    """

    sorted_sc = sorted(score_dict.values())
    gamma = np.quantile(sorted_sc, quantile)
    print(f" gamma = {gamma}")

    filtered_samples = list()
    filtered_weights = list()

    for i, s in enumerate(samples):
        # Remove failed dockings
        try:
            score = score_dict[s]
        except KeyError:
            continue

        # Drop invalid and correct smiles to kekule format to avoid reencoding issues when training search model
        # Removed not decodable molecules
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue

        # get rid of all samples with weight 0 (do not count in CbAS loss)
        # TODO : add threshold
        oracle_proba = deterministic_one(score, gamma)
        if oracle_proba == 1:
            continue

        weight = weights[i] * (1 - oracle_proba)
        filtered_samples.append(s)
        filtered_weights.append(weight)

    print(f'{len(filtered_samples)}/{len(samples)} samples kept')
    return samples, weights


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--json_path', type=str)  # the gentrain model
    parser.add_argument('--iteration', type=int, default=0)  #
    parser.add_argument('--search_name', type=str, default='search_vae')  # the prior VAE (pretrained)

    # =======
    args, _ = parser.parse_known_args()

    # print('iteration is ', args.iteration)

    # Aggregate docking results
    score_dict = gather_scores(args.iteration)
    # df = pd.read_csv(script_dir, 'results/docking_results/0.csv')
    # mol, score = df['smile'], df['score']
    # score_dict = dict(zip(mol, score))

    # Reweight and discard wrong samples
    dump_path = os.path.join(script_dir, 'results/samples.p')
    samples, weights = pickle.load(open(dump_path, 'rb'))
    samples, weights = process_samples(score_dict, samples, weights, quantile=0.5)

    # Load an instance of previous model
    search_model = model_from_json(args.prior_name)

    # Retrieve the gentrain object and feed it with updated model
    dumper = Dumper(default_model=False)
    json_path = os.path.join(script_dir, 'results', 'models', args.search_name, 'params_gentrain.json')
    params = dumper.load(json_path)
    savepath = os.path.join(params['savepath'], 'weights.pth')
    search_model.load(savepath)

    # Update search model
    search_trainer = GenTrain(search_model, **params)

    search_trainer.step('smiles', samples, weights)
