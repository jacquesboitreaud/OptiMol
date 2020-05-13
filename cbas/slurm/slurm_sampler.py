import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse

from cbas.gen_prob import GenProb
from utils import *
from model import model_from_json

import pickle


def get_samples(prior_model, search_model, max):
    sample_selfies = []
    weights = []
    sample_selfies_set = set()
    tries = 0
    stop = 100
    batch_size = 100

    # Importance weights
    while tries < stop or len(sample_selfies) < max:
        new_ones = 0

        # Get raw samples
        samples_z = search_model.sample_z_prior(n_mols=batch_size)
        gen_seq = search_model.decode(samples_z)
        _, sample_indices = torch.max(gen_seq, dim=1)

        # Compute weights while we have indices and store them: p(x|z, theta)/p(x|z, phi)
        batch_weights = GenProb(sample_indices, samples_z, prior_model) / \
                        GenProb(sample_indices, samples_z, search_model)

        # Check the novelty
        batch_selfies = search_model.indices_to_smiles(sample_indices)
        for i, s in enumerate(batch_selfies):
            new_selfie = decoder(s)
            if new_selfie not in sample_selfies_set:
                new_ones += 1
                sample_selfies_set.add(new_selfie)
                sample_selfies.append(new_selfie)
                weights.append(batch_weights[i])
        tries += 1

        print(f'{new_ones}/{batch_size} unique smiles sampled')
        weights = torch.cat(weights, dim=0)
    return sample_selfies, weights


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--search_path', type=str)  # the prior VAE (pretrained)
    parser.add_argument('--max', type=int, default=1000)  # the prior VAE (pretrained)

    # =======

    args = parser.parse_known_args()
    prior_model = model_from_json(args.prior_name)

    search_model = model_from_json(args.prior_name)
    search_model.load(args.search_path)

    samples, weights = get_samples(prior_model, search_model, max=args.max)
    dump_path = os.path.join(script_dir, 'results/samples.p')
    pickle.dump((samples, weights), open(dump_path, 'wb'))

    dumper = Dumper(default_model=False)
    params = dumper.load(args.json_path)
    savepath = params['savepath']
