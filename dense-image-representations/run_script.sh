#!/bin/bash
#SBATCH --job-name=2565e-05
#SBATCH --output results/12-clip/train.log
#SBATCH --error results/12-clip/train.log
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
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 20 \
--projection_dim 512 \
--lr 5e-05 \
--num_heads 8 \
--num_layers 12 \
--text_encoder clip \
--exp_name 12-clip \