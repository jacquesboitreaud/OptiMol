#!/bin/sh
#SBATCH --job-name=sampler
#SBATCH --output=out_slurm/sampler_%A.out
#SBATCH --error=out_slurm/sampler_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate optimol_cpu

python sampler.py --prior_name $1 --search_name $2 --max_samples $3 $4


