import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse
import pickle
import pandas as pd

from cbas.gen_train import GenTrain
from cbas.oracles import deterministic_cdf_oracle
from model import model_from_json
from utils import *


def gather_scores(iteration):
    """
    Gather docking results
    :return:
    """
    dirname = os.path.join(script_dir, 'results', 'docking_small_results')
    dfs = [pd.read_csv(csv_file) for csv_file in os.listdir(dirname)]
    merged = pd.concat(dfs)
    dump_path = os.path.join(script_dir, 'results', f'docking_results_{iteration}')
    merged.to_csv(dump_path)

    molecules = merged['smiles']
    scores = merged['smiles']

    score_dict = dict(zip(molecules, scores))
    return score_dict


def process_samples(score_dict, samples, weights):
    """
    reweight samples
    :return:
    """
    # We need to rearrange the order after broadcasting
    scores = [score_dict[molecule] for molecule in samples]

    sorted_sc = sorted(scores)
    gamma = np.quantile(sorted_sc, args.Q)
    print(f"step {t}/{args.iters}, gamma = {gamma}")

    # Weight samples
    scores = np.array(scores)

    # Update weights by proba that oracle passes threshold
    weights = weights * (1 - deterministic_cdf_oracle(scores, gamma))  # weight 0 if oracle < gamma

    # Drop invalid and correct smiles to kekule format to avoid reencoding issues when training search model
    good_indices = []
    for i, s in enumerate(samples):
        m = Chem.MolFromSmiles(s)
        if m is not None and weights[i] > 0:  # get rid of all samples with weight 0 (do not count in CbAS loss)
            good_indices.append(i)

    samples = [samples[i] for i in good_indices]
    weights = weights[good_indices]

    print(f'{len(good_indices)}/{args.M} samples kept')
    return samples, weights


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--search_path', type=str)  # the prior VAE (pretrained)
    parser.add_argument('--iteration', type=int, default=0)  # the prior VAE (pretrained)

    # =======
    args = parser.parse_known_args()

    # Aggregate docking results
    score_dict = gather_scores(args.iteration)

    # Reweight and discard wrong samples
    dump_path = os.path.join(script_dir, 'results/samples.p')
    samples, weights = pickle.load(open(dump_path, 'rb'))
    samples, weights = process_samples(score_dict, samples, weights)

    search_model = model_from_json(args.prior_name)
    dumper = Dumper(default_model=False)
    params = dumper.load(args.json_path)
    savepath = params['savepath']
    search_model.load(savepath)

    # Update search model
    search_trainer = GenTrain(search_model, **params)
    search_trainer.step('smiles', samples, weights)
