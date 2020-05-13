#!/bin/sh
#SBATCH --job-name=sampler
#SBATCH --output=out_slurm/sampler_%A.out
#SBATCH --error=out_slurm/sampler_%A.err

source ~/anaconda3/bin/activate
conda activate optimol

python sampler.py