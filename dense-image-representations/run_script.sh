#!/bin/bash
#SBATCH --job-name=comp
#SBATCH --output launch.log
#SBATCH --error launch.log
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

python3 compute_vision_graphs.py \
--visual_graphs_save_path coco_visual_graph \
--batch_size 1 \