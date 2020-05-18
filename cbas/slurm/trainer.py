import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse
import pickle
import pandas as pd
import shutil

from cbas.gen_train import GenTrain
from cbas.oracles import deterministic_one, normal_cdf_oracle
from model import model_from_json
from utils import *


def gather_scores(iteration, name):
    """
    Gather docking results from small csv and merge them in a big iteration.csv file.
    :param iteration: integer to index the merged csv
    :param name: to get the loading and dumping path
    :return: {smile : docking_score}
    """
    dirname = os.path.join(script_dir, 'results', name, 'docking_small_results')
    dfs = [pd.read_csv(os.path.join(dirname, csv_file)) for csv_file in os.listdir(dirname)]
    merged = pd.concat(dfs)
    dump_path = os.path.join(script_dir, 'results', name, 'docking_results', f'{iteration}.csv')
    merged.to_csv(dump_path)

    def empty_folder(folder_path):
        for file_object in os.listdir(folder_path):
            file_object_path = os.path.join(folder_path, file_object)
            if os.path.isfile(file_object_path) or os.path.islink(file_object_path):
                os.unlink(file_object_path)
            else:
                shutil.rmtree(file_object_path)

    empty_folder(dirname)

    molecules = merged['smile']
    scores = merged['score']

    return dict(zip(molecules, scores))


def process_samples(score_dict, samples, weights, quantile, oracle='binary', threshold=0.05, qed=False):
    """
    reweight samples using docking scores
    :return:
    """
    # We maximize an objective but cbas actually minimizes things
    if not qed:
        for key, value in score_dict.items():
            score_dict[key] = -value
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

        # Oracle proba is p(score<=gamma) so (1 - oracle_proba) =  p(score>gamma)
        # Get rid of all samples with too small weight (do not count in CbAS loss)
        # For a gaussian, default 0.05 is approx 1.6 stds so we take into account docking scores up until :
        # (quantile - 1.6std) with weight 0.05

        if oracle == 'binary':
            oracle_proba = deterministic_one(score, gamma)
        elif oracle == 'gaussian':
            oracle_proba = normal_cdf_oracle(score, gamma)
        else:
            raise ValueError('wrong option')
        # print(f'p(score>gamma = {(1 - oracle_proba)}, and threshold is {threshold}')
        if (1 - oracle_proba) < threshold:
            continue
        weight = weights[i] * (1 - oracle_proba)
        filtered_samples.append(s)
        filtered_weights.append(weight)

    print(f'{len(filtered_samples)}/{len(samples)} samples kept')
    return filtered_samples, filtered_weights


def main(iteration, quantile, oracle, prior_name, name, qed):
    # Aggregate docking results
    score_dict = gather_scores(iteration, name)

    # Memoization of the sampled compounds, if they are not qed scores
    if not qed:
        print('doing memoization')
        whole_path = os.path.join(script_dir, '..', '..', 'data', 'drd3_scores.pickle')
        docking_whole_results = pickle.load(open(whole_path, 'rb'))
        docking_whole_results.update(score_dict)
        pickle.dump(docking_whole_results, open(whole_path, 'wb'))

    # Reweight and discard wrong samples
    dump_path = os.path.join(script_dir, 'results', name, 'samples.p')
    samples, weights = pickle.load(open(dump_path, 'rb'))
    samples, weights = process_samples(score_dict, samples, weights, oracle=oracle, quantile=quantile, qed=qed)

    # Load an instance of previous model
    search_model = model_from_json(prior_name)

    # Retrieve the gentrain object and feed it with updated model
    dumper = ModelDumper(default_model=False)
    json_path = os.path.join(script_dir, 'results', name, 'params_gentrain.json')
    params = dumper.load(json_path)
    savepath = os.path.join(params['savepath'], 'weights.pth')
    search_model.load(savepath)

    # Update search model
    search_trainer = GenTrain(search_model, **params)
    search_trainer.step('smiles', samples, weights)


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--name', type=str, default='search_vae')  # the experiment name
    parser.add_argument('--quantile', type=float, default=0.6)  # quantile of scores accepted
    parser.add_argument('--oracle', type=str, default='gaussian')  # the mode of the oracle
    parser.add_argument('--qed', action='store_true')

    args, _ = parser.parse_known_args()

    main(iteration=args.iteration,
         quantile=args.quantile,
         oracle=args.oracle,
         prior_name=args.prior_name,
         name=args.name,
         qed=args.qed)
