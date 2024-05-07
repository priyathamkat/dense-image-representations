#!/bin/bash
#SBATCH --job-name=2565e-05
#SBATCH --output results/clip-clip_baseline/train.log
#SBATCH --error results/clip-clip_baseline/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

python3 contrastive_train_baseline.py \
--text_tokens_train coco_text_tokens \
--text_tokens_val coco_val_text_tokens \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 20 \
--projection_dim 512 \
--image_encoder clip \
--text_encoder clip \
--lr 5e-05 \
--exp_name clip-clip_baseline \