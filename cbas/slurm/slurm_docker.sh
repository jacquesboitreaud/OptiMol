#!/bin/sh
#SBATCH --job-name=docker
#SBATCH --output=out_slurm/docker_%A.out
#SBATCH --error=out_slurm/docker_%A.err
#SBATCH --array=0-199

# This is a jobarray file that can be run from master

source ~/anaconda3/etc/profile.d/conda.sh
conda activate optimol_cpu

python docker.py $SLURM_ARRAY_TASK_ID 200 --server $1 --exhaustiveness $2 --name $3 --oracle $4

