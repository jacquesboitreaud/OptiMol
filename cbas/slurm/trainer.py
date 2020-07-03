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
    dict_scores = dict(zip(molecules, scores))
    return {mol: score for mol, score in dict_scores.items() if score != 0}


def process_samples(score_dict, samples, weights, quantile, prev_gamma=-1000, uncertainty='binary', threshold=0.05,
                    oracle='qed'):
    """
    reweight samples using docking scores
    :return:
    """
    # We maximize an objective but for docking we actually need to minimize things
    if oracle == 'docking':
        for key, value in score_dict.items():
            score_dict[key] = -value
    sorted_sc = sorted(score_dict.values())

    new_gamma = np.quantile(sorted_sc, quantile)
    gamma = new_gamma if new_gamma > prev_gamma else prev_gamma
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

        if uncertainty == 'binary':
            oracle_proba = deterministic_one(score, gamma)
        elif uncertainty == 'gaussian':
            std = 0.03 if oracle == 'qed' else 1
            oracle_proba = normal_cdf_oracle(score, gamma, std=std)
        else:
            raise ValueError('wrong option')
        # print(f'p(score>gamma = {(1 - oracle_proba)}, and threshold is {threshold}')
        if (1 - oracle_proba) < threshold:
            continue
        weight = weights[i] * (1 - oracle_proba)
        filtered_samples.append(s)
        filtered_weights.append(weight)

    print(f'{len(filtered_samples)}/{len(samples)} samples kept')
    return filtered_samples, filtered_weights, gamma


def main(iteration, quantile, uncertainty, prior_name, name, oracle):
    # Aggregate docking results using previous gamma
    score_dict = gather_scores(iteration, name)

    # Memoization of the sampled compounds, if they are docking scores
    if oracle == 'docking':
        print('doing memoization')
        whole_path = os.path.join(script_dir, '..', '..', 'data', 'drd3_scores.pickle')
        docking_whole_results = pickle.load(open(whole_path, 'rb'))
        # Only update memoization for successful dockings
        new_results = {key: value for key, value in score_dict.items() if value < 0}
        docking_whole_results.update(new_results)
        pickle.dump(docking_whole_results, open(whole_path, 'wb'))

    # Reweight and discard wrong samples
    dump_path = os.path.join(script_dir, 'results', name, 'samples.p')
    samples, weights = pickle.load(open(dump_path, 'rb'))

    dumper = Dumper()
    json_path = os.path.join(script_dir, 'results', name, 'params_gentrain.json')
    params = dumper.load(json_path)
    gamma = params['gamma']

    samples, weights, gamma = process_samples(score_dict, samples, weights, uncertainty=uncertainty, quantile=quantile,
                                              oracle=oracle, prev_gamma=gamma)
    params['gamma'] = gamma
    dumper.dump(dict_to_dump=params, dumping_path=json_path)
    params.pop('gamma')

    # Load an instance of previous model
    search_model = model_from_json(prior_name)

    # Retrieve the gentrain object and feed it with updated model
    savepath = os.path.join(params['savepath'], 'weights.pth')
    search_model.load(savepath)
    search_trainer = GenTrain(search_model, **params)

    # send to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    search_model.to(device)
    search_trainer.device = device
    search_trainer.load_optim()

    # Update search model
    search_trainer.step('smiles', samples, weights)

    # Add model dumping at each epoch
    weights_path = os.path.join(search_trainer.savepath, f"weights_{iteration}.pth")
    torch.save(search_trainer.model.state_dict(), weights_path)


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--iteration', type=int, default=0)
    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--name', type=str, default='search_vae')  # the experiment name
    parser.add_argument('--quantile', type=float, default=0.6)  # quantile of scores accepted
    parser.add_argument('--uncertainty', type=str, default='gaussian')  # the mode of the oracle
    parser.add_argument('--oracle', type=str)  # 'qed' or 'docking' or 'qsar'

    args, _ = parser.parse_known_args()

    main(prior_name=args.prior_name,
         name=args.name,
         iteration=args.iteration,
         quantile=args.quantile,
         uncertainty=args.uncertainty,
         oracle=args.oracle)
