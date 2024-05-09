#!/bin/bash
#SBATCH --job-name=2561e-06
#SBATCH --output results_clip32/1e-06-clip_clip_transformer/train.log
#SBATCH --error results_clip32/1e-06-clip_clip_transformer/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

python3 contrastive_train.py \
--dataset coco \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 20 \
--projection_dim 512 \
--lr 1e-06 \
--num_heads 8 \
--num_layers 6 \
--text_encoder clip \
--exp_name 1e-06-clip_clip_transformer \