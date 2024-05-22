#!/bin/bash
#SBATCH --job-name=2561e-05
#SBATCH --output results_clip_edge_tokens/1e-05-vit_s-clip_baseline/train.log
#SBATCH --error results_clip_edge_tokens/1e-05-vit_s-clip_baseline/train.log
#SBATCH --time=70:00:00
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

accelerate launch --multi_gpu \
--main_process_port 37789 \
contrastive_train_baseline.py \
--dataset coco \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 5 \
--projection_dim 512 \
--image_encoder vit_s \
--text_encoder clip \
--lr 1e-05 \
--exp_name 1e-05-vit_s-clip_baseline \
--result_dir results_clip_edge_tokens \