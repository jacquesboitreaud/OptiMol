#!/bin/sh
#SBATCH --account=def-jeromew
#SBATCH --time=72:00:00

#SBATCH --job-name=run_array
#SBATCH --output=out_slurm/run-array_%A.out
#SBATCH --error=out_slurm/run-array_%A.err
#SBATCH --array=0-9

python3 docking.py $SLURM_ARRAY_TASK_ID 10 -s cedar