#!/bin/bash
#SBATCH --job-name=2561e-06
#SBATCH --output results_clip_edge_tokens/1e-06-8-clip_preembed_attn_mask/train.log
#SBATCH --error results_clip_edge_tokens/1e-06-8-clip_preembed_attn_mask/train.log
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:rtxa5000:2
#SBATCH --qos=scavenger
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

accelerate launch --multi_gpu \
--main_process_port 30780 \
contrastive_train.py \
--dataset coco \
--batch_size 256 \
--epochs 100 \
--warmup 10 \
--validation_epochs 5 \
--checkpoint_epochs 20 \
--projection_dim 512 \
--lr 1e-06 \
--num_heads 8 \
--num_layers 8 \
--text_encoder clip \
--transformer ours \
--result_dir results_clip_edge_tokens \
--exp_name 1e-06-8-clip_preembed_attn_mask \
--use_attention_mask \
--preembed_nodes \