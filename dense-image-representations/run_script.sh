#!/bin/bash
#SBATCH --job-name=comp
#SBATCH --output launch.log
#SBATCH --error launch.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

python compute_vision_graphs.py \
--dataset winoground \
--visual_graphs_save_path winoground_visual_graph \
--parsed_captions_path winoground_parsed_captions \
--batch_size 1 \