#!/bin/sh
#SBATCH --job-name=run_array
#SBATCH --output=out_slurm/trainer_%A.out
#SBATCH --error=out_slurm/trainer_%A.err
#SBATCH --array=0-59

# This is a jobarray file that can be run from master

python trainer.py $2