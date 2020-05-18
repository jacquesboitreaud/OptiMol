#!/bin/sh
#SBATCH --job-name=trainer
#SBATCH --output=out_slurm/trainer_%A.out
#SBATCH --error=out_slurm/trainer_%A.err
#SBATCH -p c7gpu
#SBATCH --gres=gpu:1

module load cuda/10.1.243_418.87.00
module load cudnn/v7.6.5.32/cuda-10.1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate optimol

python trainer.py --prior_name $1 --name $2 --iteration $3 --quantile $4 --oracle $5 $6


