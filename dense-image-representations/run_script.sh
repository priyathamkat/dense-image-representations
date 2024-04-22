#!/bin/bash
#SBATCH --job-name=2561e-05
#SBATCH --output results/coco-256-1e-05-1000/train.log
#SBATCH --error results/coco-256-1e-05-1000/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

python3 contrastive_train.py \
--vision_tokens_train coco_visual_tokens \
--text_tokens_train coco_text_tokens \
--vision_tokens_val coco_val_visual_tokens \
--text_tokens_val coco_val_text_tokens \
--batch_size 512 \
--epochs 500 \
--warmup 30000 \
--validation_epochs 20 \
--checkpoint_epochs 20 \
--lr 5e-05 \
--num_heads 8 \
--num_layers 6 \
--projection_dim 128 \
--exp_name coco-512-5e-05-500 \