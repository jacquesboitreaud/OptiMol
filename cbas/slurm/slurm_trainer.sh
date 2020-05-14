#!/bin/sh
#SBATCH --job-name=trainer
#SBATCH --output=out_slurm/trainer_%A.out
#SBATCH --error=out_slurm/trainer_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate optimol_cpu

python trainer.py --iteration $1