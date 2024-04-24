#!/bin/bash
#SBATCH --job-name=2561e-06
#SBATCH --output results/coco-8-6-256-1e-06/train.log
#SBATCH --error results/coco-8-6-256-1e-06/train.log
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

python3 winoground_eval.py \
--exp_name coco-512-5e-05-500 \