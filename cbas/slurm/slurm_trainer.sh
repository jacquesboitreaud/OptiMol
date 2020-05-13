#!/bin/sh
#SBATCH --job-name=sampler
#SBATCH --output=out_slurm/trainer_%A.out
#SBATCH --error=out_slurm/trainer_%A.err

source ~/anaconda3/bin/activate
conda activate optimol

python trainer.py $2