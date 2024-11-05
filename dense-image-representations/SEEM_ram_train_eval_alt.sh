#!/bin/bash
#SBATCH --job-name=sumit-ram-train-alt
#SBATCH --output ram_train_alt.log
#SBATCH --error ram_train_alt.error.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=cml-medium
#SBATCH --account=cml-sfeizi
#SBATCH --partition=cml-dpart
#SBATCH --mem=64G
#SBATCH --nodes=1

cd /cmlscratch/snawathe/dense-image-representations/dense-image-representations

srun --nodes=1 --mem=64G /nfshomes/snawathe/micromamba/envs/dense-image-representations/bin/python ./SEEM_ram_train_eval_alt.py

wait