#!/bin/bash
#SBATCH --job-name=2560.01
#SBATCH --output results/coco-32-2-256-0.01.log
#SBATCH --error results/coco-32-2-256-0.01.log
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

python3 train.py \
--vision_graph_data coco_visual_graph \
--text_graph_data coco_parsed_captions \
--batch_size 256 \
--hidden_channels 32 \
--num_layers 2 \
--lr 0.01 \
--exp_name coco-32-2-256-0.01 \