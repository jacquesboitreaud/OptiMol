#!/bin/sh
#SBATCH --job-name=run_array
#SBATCH --output=out_slurm/docker_%A.out
#SBATCH --error=out_slurm/docker_%A.err
#SBATCH --array=0-59

# This is a jobarray file that can be run from master

python docker.py $SLURM_ARRAY_TASK_ID 60 -s pasteur