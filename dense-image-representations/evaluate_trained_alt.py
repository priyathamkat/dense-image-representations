import os
import time
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
import wandb

from mmengine.config import Config
from mmengine.utils import ProgressBar
from transformers import AutoConfig, AutoModel

from SEEM_ram_train_eval_alt import RamDataset, RamModel, RamPredictor
import pickle as pkl



config = dict(
	dataset=dict(
		data_path="/cmlscratch/snawathe/dense-image-representations/ram_dataset_full.npz",
		is_train=False,
		num_relation_classes=56,
	),
	dataloader=dict(
		batch_size=4,
	),
	model=dict(
		pretrained_model_name_or_path="bert-base-uncased",
		load_pretrained_weights=True,
		num_transformer_layer=2,
		input_feature_size=512,
		output_feature_size=768,
		cls_feature_size=512,
		num_relation_classes=56,
		pred_type="attention",
		loss_type="multi_label_ce",
	),
	optim=dict(
		lr=1e-4,
		weight_decay=0.05,
		eps=1e-8,
		betas=(0.9, 0.999),
		max_norm=0.01,
		norm_type=2,
		lr_scheduler=dict(
			step=[6, 10],
			gamma=0.1,
		),
	),
	num_epoch=100,
	output_dir="/cmlscratch/snawathe/dense-image-representations/training_output_alt_post",
	load_from=None,
)
config = Config(config)



lst = []
for i in range(20):
	config.load_from = f"/cmlscratch/snawathe/dense-image-representations/training_output_alt/epoch_{i+1}.pth"
	predictor = RamPredictor(config)
	metric = predictor.eval()
	print(f"{metric=}")
	lst.append(metric)
	print(f"Done {i+1=}")

with open("/cmlscratch/snawathe/dense-image-representations/dense-image-representations/evaluate_trained_alt_values.pkl", "wb") as f:
	pkl.dump(lst, f)