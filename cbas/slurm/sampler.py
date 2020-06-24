import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == '__main__':
    sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse

from cbas.gen_prob import GenProb
from utils import *
from model import model_from_json

import pickle

# rdkit for diversity picker 
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit import DataStructs
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker


def get_samples(prior_model, search_model, max):
    """
    Take initial samples from a prior model. Computes importance sampling weights
    This will try to produce new ones up until a certain limit of tries is reached
    :param prior_model:
    :param search_model:
    :param max:
    :return:
    """
    sample_selfies = []
    weights = []
    sample_selfies_set = set()
    tries = 0
    batch_size = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    search_model.to(device)
    prior_model.to(device)

    # Importance weights
    while (tries * batch_size) < (10 * max) and len(sample_selfies) < max:
        tries += 1

        # Get raw samples
        samples_z = search_model.sample_z_prior(n_mols=batch_size)
        gen_seq = search_model.decode(samples_z)
        _, sample_indices = torch.max(gen_seq, dim=1)

        # Compute weights while we have indices and store them: p(x|z, theta)/p(x|z, phi)
        prior_prob = GenProb(sample_indices, samples_z, prior_model)
        search_prob = GenProb(sample_indices, samples_z, search_model)
        batch_weights = torch.exp(prior_prob - search_prob)

        # Check the novelty
        new_ones = 0
        batch_selfies = search_model.indices_to_smiles(sample_indices)
        for i, s in enumerate(batch_selfies):
            new_selfie = decoder(s)
            if new_selfie not in sample_selfies_set:
                new_ones += 1
                sample_selfies_set.add(new_selfie)
                sample_selfies.append(new_selfie)
                weights.append(batch_weights[i])

        print(f'{tries} : {new_ones}/{batch_size} unique smiles sampled')
    return sample_selfies, weights


def main(prior_name, name, max_samples, diversity_picker, oracle):
    prior_model = model_from_json(prior_name)

    # We start by creating another prior instance, then replace it with the actual weights
    # name = search_vae
    search_model = model_from_json(prior_name)
    model_weights_path = os.path.join(script_dir, 'results', name, 'weights.pth')
    search_model.load(model_weights_path)

    samples, weights = get_samples(prior_model, search_model, max=max_samples)
    
    # if diversity picker < max_samples, we subsample with rdkit picker : 
    if diversity_picker < max_samples : 
    
        mols = [Chem.MolFromSmiles(s) for s in samples]
        fps = [GetMorganFingerprint(x,3) for x in mols]
        picker = MaxMinPicker()
        
        def distij(i,j,fps=fps):
            return 1-DataStructs.DiceSimilarity(fps[i],fps[j])
        
        pickIndices = picker.LazyPick(distij,max_samples,diversity_picker)
        idces = list(pickIndices)
        samples = [samples[i] for i in idces]
        weights = [weights[i] for i in idces]

    # Since we don't maintain a dict for qed, we just give everything to the docker
    if oracle != 'docking':
        dump_path = os.path.join(script_dir, 'results', name, 'docker_samples.p')
        pickle.dump(samples, open(dump_path, 'wb'))

        # Dump for the trainer
        dump_path = os.path.join(script_dir, 'results', name, 'samples.p')
        pickle.dump((samples, weights), open(dump_path, 'wb'))

    else:
        # Memoization, we split the list into already docked ones and dump a simili-docking csv
        whole_path = os.path.join(script_dir, '..', '..', 'data', 'drd3_scores.pickle')
        docking_whole_results = pickle.load(open(whole_path, 'rb'))
        filtered_smiles = list()
        already_smiles = list()
        already_scores = list()
        for i, smile in enumerate(samples):
            if smile in docking_whole_results:
                already_smiles.append(smile)
                already_scores.append(docking_whole_results[smile])
            else:
                filtered_smiles.append(smile)

        # Dump simili-docking
        dump_path = os.path.join(script_dir, 'results', name, 'docking_small_results', 'simili.csv')
        df = pd.DataFrame.from_dict({'smile': already_smiles, 'score': already_scores})
        df.to_csv(dump_path)

        # Dump for the docker
        dump_path = os.path.join(script_dir, 'results', name, 'docker_samples.p')
        pickle.dump(filtered_smiles, open(dump_path, 'wb'))

        # Dump for the trainer
        dump_path = os.path.join(script_dir, 'results', name, 'samples.p')
        pickle.dump((samples, weights), open(dump_path, 'wb'))


if __name__ == '__main__':
    pass

    parser = argparse.ArgumentParser()

    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('--name', type=str, default='search_vae')  # the prior VAE (pretrained)
    parser.add_argument('--max_samples', type=int, default=1000)  # samples drawn from model
    parser.add_argument('--diversity_picker', type=int, default=100)  # diverse samples subset size. if = max_samples, all selected
    parser.add_argument('--oracle', type=str)  # 'qed' or 'docking' or 'qsar'
    # =======

    args, _ = parser.parse_known_args()

    main(prior_name=args.prior_name,
         name=args.name,
         max_samples=args.max_samples,
         diversity_picker = args.diversity_picker,
         oracle=args.oracle)
