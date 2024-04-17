#!/bin/bash
#SBATCH --job-name=2560.0001
#SBATCH --output results/coco-8-6-256-0.0001/train.log
#SBATCH --error results/coco-8-6-256-0.0001/train.log
#SBATCH --time=27:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

python3 contrastive_train.py \
--vision_tokens coco_visual_tokens \
--text_tokens coco_text_tokens \
--batch_size 256 \
--lr 0.0001 \
--num_heads 8 \
--num_layers 6 \
--projection_dim 128 \
--exp_name coco-8-6-256-0.0001 \