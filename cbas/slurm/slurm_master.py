"""

Slurm Master

Params to be blended/changed :


"""
import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', '..'))

import argparse
import subprocess
import torch

from utils import Dumper, soft_mkdir
from model import model_from_json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prior_name', type=str, default='inference_default')  # the prior VAE (pretrained)
    parser.add_argument('-n', '--name', type=str, default='search_vae')  # the name of the experiment
    parser.add_argument('--iters', type=int, default=2)  # Number of iterations
    parser.add_argument('--oracle', type=str, default='qed')  # 'qed' or 'docking' or 'qsar'

    # SAMPLER
    parser.add_argument('--max_samples', type=int, default=3000)  # Nbr of samples at each iter
    parser.add_argument('--cap_weights', type=float, default=-1)  # clamping value to use

    # DOCKER
    parser.add_argument('--server', type=str, default='pasteur', help='server to run on')  # the prior VAE (pretrained)
    parser.add_argument('--ex', type=int, default=16)  # Nbr of samples at each iter

    # TRAINER
    parser.add_argument('--quantile', type=float, default=0.6)  # quantile of scores accepted
    parser.add_argument('--uncertainty', type=str, default='gaussian')  # the mode of the oracle

    # GENTRAIN
    parser.add_argument('--procs', type=int, default=0)  # Number of processes for VAE dataloading
    parser.add_argument('--epochs', type=int, default=5)  # Number of iterations
    parser.add_argument('--learning_rate', type=float, default=1e-4)  # Number of iterations
    parser.add_argument('--beta', type=float, default=0.2)  # KL weight in loss function
    parser.add_argument('--clip_grad_norm', type=float, default=5.0)  # quantile of scores accepted
    parser.add_argument('--opti', type=str, default='adam')  # the mode of the oracle
    parser.add_argument('--sched', type=str, default='elr')  # the mode of the oracle
    parser.add_argument('--alphabet_name', type=str, default='fabritiis.json')  # the alphabet used
    # =======

    args, _ = parser.parse_known_args()
    device = 'cpu'  # 'cuda' if torch.cuda.is_available() else 'cpu'

    assert args.oracle in ['qed', 'docking', 'qsar']


    def setup():
        pass
        soft_mkdir(os.path.join(script_dir, 'results'))
        soft_mkdir(os.path.join(script_dir, 'results', args.name))
        soft_mkdir(os.path.join(script_dir, 'results', args.name, 'docking_results'))
        soft_mkdir(os.path.join(script_dir, 'results', args.name, 'docking_small_results'))


    setup()
    savepath = os.path.join(script_dir, 'results', args.name)
    soft_mkdir(savepath)

    # Save experiment parameters
    dumper = Dumper(dumping_path=os.path.join(savepath, 'experiment.json'), dic=args.__dict__)
    dumper.dump()

    params_gentrain = {'savepath': savepath,
                       'epochs': args.epochs,
                       'device': device,
                       'lr': args.learning_rate,
                       'clip_grad': args.clip_grad_norm,
                       'beta': args.beta,
                       'processes': args.procs,
                       'optimizer': args.opti,
                       'scheduler': args.sched,
                       'alphabet_name': args.alphabet_name,
                       'gamma': -1000,
                       'DEBUG': True}
    dumper = Dumper(dumping_path=os.path.join(savepath, 'params_gentrain.json'), dic=params_gentrain)
    dumper.dump()

    prior_model_init = model_from_json(args.prior_name)
    print(prior_model_init)
    torch.save(prior_model_init.state_dict(), os.path.join(savepath, "weights.pth"))
    id_train = None

    for iteration in range(1, args.iters + 1):
        # SAMPLING
        slurm_sampler_path = os.path.join(script_dir, 'slurm_sampler.sh')
        if id_train is None:
            cmd = f'sbatch {slurm_sampler_path}'
        else:
            cmd = f'sbatch --depend=afterany:{id_train} {slurm_sampler_path}'
        extra_args = f' {args.prior_name} {args.name} {args.max_samples} {args.oracle} {args.cap_weights}'
        cmd = cmd + extra_args
        a = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        id_sample = a.split()[3]

        # DOCKING
        slurm_docker_path = os.path.join(script_dir, 'slurm_docker.sh')
        cmd = f'sbatch --depend=afterany:{id_sample} {slurm_docker_path}'
        extra_args = f' {args.server} {args.ex} {args.name} {args.oracle}'
        cmd = cmd + extra_args
        a = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        id_dock = a.split()[3]

        # AGGREGATION AND TRAINING
        slurm_trainer_path = os.path.join(script_dir, 'slurm_trainer.sh')
        cmd = f'sbatch --depend=afterany:{id_dock} {slurm_trainer_path}'
        extra_args = f' {args.prior_name} {args.name} {iteration} {args.quantile} {args.uncertainty} {args.oracle}'
        cmd = cmd + extra_args
        a = subprocess.run(cmd.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        id_train = a.split()[3]

        print(f'launched iteration {iteration}')
