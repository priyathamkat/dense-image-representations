#!/bin/bash
#SBATCH --job-name=2565e-05
#SBATCH --output results_clip32/5e-05-12-clip/train.log
#SBATCH --error results_clip32/5e-05-12-clip/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

accelerate launch --multi_gpu \
--main_process_port 17320 \
contrastive_train.py \
--dataset coco \
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
--transformer ours \
--result_dir results_clip32 \
--exp_name 5e-05-12-clip \