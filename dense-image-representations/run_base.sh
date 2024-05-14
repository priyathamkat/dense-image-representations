#!/bin/bash
#SBATCH --job-name=2565e-06
#SBATCH --output results_clip_edge_tokens/5e-06-vit-clip_baseline/train.log
#SBATCH --error results_clip_edge_tokens/5e-06-vit-clip_baseline/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

accelerate launch --multi_gpu \
--main_process_port 17538 \
contrastive_train_baseline.py \
--dataset coco \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 5 \
--projection_dim 512 \
--image_encoder vit \
--text_encoder clip \
--lr 5e-06 \
--exp_name 5e-06-vit-clip_baseline \
--result_dir results_clip_edge_tokens \