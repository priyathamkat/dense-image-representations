#!/bin/bash
#SBATCH --job-name=data_prep
#SBATCH --output=data_prep.log
#SBATCH --error=data_prep.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G


python3 data_processing/visual_genome.py