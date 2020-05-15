#!/bin/sh
#SBATCH --job-name=trainer
#SBATCH --output=out_slurm/trainer_%A.out
#SBATCH --error=out_slurm/trainer_%A.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate optimol_cpu

python trainer.py --prior_name $1 --name $2 --iteration $3 --quantile $4 --oracle $5 $6


