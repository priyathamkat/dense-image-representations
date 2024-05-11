#!/bin/bash
#SBATCH --job-name=2561e-06
#SBATCH --output results_clip32/1e-06-clip-clip_baseline/train.log
#SBATCH --error results_clip32/1e-06-clip-clip_baseline/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:4
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

accelerate launch --multi_gpu \
--main_process_port 17320 \
contrastive_train_baseline.py \
--dataset coco \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 5 \
--projection_dim 512 \
--image_encoder clip \
--text_encoder clip \
--lr 1e-06 \
--exp_name 1e-06-clip-clip_baseline \
--result_dir results_clip32 \