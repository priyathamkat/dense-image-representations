#!/bin/bash
#SBATCH --job-name=sumit-evaluate-alt
#SBATCH --output ram_evaluate_alt.log
#SBATCH --error ram_evaluate_alt.error.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=cml-scavenger
#SBATCH --account=cml-scavenger
#SBATCH --partition=cml-scavenger
#SBATCH --mem=64G
#SBATCH --nodes=1

cd /cmlscratch/snawathe/dense-image-representations/dense-image-representations

srun --nodes=1 --mem=64G /nfshomes/snawathe/micromamba/envs/dense-image-representations/bin/python ./evaluate_trained_alt.py

wait