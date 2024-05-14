#!/bin/bash
#SBATCH --job-name=2565e-05
#SBATCH --output results_clip_edge_tokens/5e-05-8-clip_attn_mask/train.log
#SBATCH --error results_clip_edge_tokens/5e-05-8-clip_attn_mask/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

accelerate launch --multi_gpu \
--main_process_port 30640 \
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
--num_layers 8 \
--text_encoder clip \
--transformer ours \
--result_dir results_clip_edge_tokens \
--exp_name 5e-05-8-clip_attn_mask \
--use_attention_mask \