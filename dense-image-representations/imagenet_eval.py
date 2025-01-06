import argparse 
import glob
import os
import numpy as np

import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip
from transformers import ViTImageProcessor

from data.datautils import get_dataset
import utils


from contrastive_train import forward_pass
from contrastive_train_baseline import forward_pass as forward_pass_base

from modules import VisionLanguageEncoder, VisionLanguageEncoderBase

from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--preembed_nodes', action='store_true')
    parser.add_argument('--use_attention_mask', action='store_true')

    parser.add_argument('--text_encoder', type=str, default='t5')
    parser.add_argument('--image_encoder', type=str, default='vit')
    parser.add_argument('--transformer', type=str, default='clip')


    args = parser.parse_args()

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])

    clip_model = None
    if 'clip' in [args.image_encoder, args.text_encoder]:
        clip_model, clip_image_processor = clip.load("ViT-B/32", device='cuda')
        clip_model = clip_model.to(torch.float32)

    if 'baseline' in args.exp_name:
        vision_language_encoder = VisionLanguageEncoderBase(projection_dim=args.projection_dim,
                                                            text_encoder=args.text_encoder,
                                                            image_encoder=args.image_encoder,
                                                            clip_model=clip_model,)

        if 'vit_small' in args.image_encoder:
            image_processor = ViTImageProcessor.from_pretrained('facebook/dino-vits16')
        elif 'vit' in args.image_encoder:
            image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
        else:
            image_processor = clip_image_processor

        dataset = get_dataset(
            dataset_name = 'imagenet_val',
            transform = image_processor,
            with_image_tokens = False, 
            caption_return_policy = 'all',
            hf_vit_processor = 'vit' in args.image_encoder,
        )
        forward_pass_method = forward_pass_base

    else:
        dataset = get_dataset(
            dataset_name = 'imagenet_val',
            image_tokens_root = f'imagenet_val_visual_tokens',
            with_image_tokens = True, 
            caption_return_policy = 'all'
        )

        vision_language_encoder = VisionLanguageEncoder(projection_dim=args.projection_dim,
                                                        transformer_width=768 if args.transformer == 'clip' else 512, 
                                                        transformer_heads=args.num_heads, 
                                                        transformer_layers=args.num_layers,
                                                        image_embedding_size=2880,
                                                        preembed_nodes=args.preembed_nodes,
                                                        text_encoder=args.text_encoder,
                                                        clip_model=clip_model,
                                                        transformer=args.transformer, 
                                                        use_attention_mask=args.use_attention_mask,)
        forward_pass_method = forward_pass

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    checkpoint_dir = f'{args.result_dir}/{args.exp_name}'
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/model_*"), key=os.path.getmtime, reverse=True)
    if len(ckpts) == 0:
        print(f"No checkpoints found in {checkpoint_dir}")
    else:
        print(f"Loading state {ckpts[0]}")
        if '.pth.tar' in ckpts[0]:
            vision_language_encoder = vision_language_encoder.cuda()
            vision_language_encoder = nn.DataParallel(vision_language_encoder)
            state = torch.load(ckpts[0])['state_dict']
            vision_language_encoder.load_state_dict(state)
            vision_language_encoder = vision_language_encoder.module
        else:
            device = accelerator.device
            vision_language_encoder = vision_language_encoder.to(device)
            vision_language_encoder = accelerator.prepare(vision_language_encoder)
            loader = accelerator.prepare(loader)
            accelerator.wait_for_everyone()
            accelerator.load_state(input_dir=checkpoint_dir)
            vision_language_encoder = vision_language_encoder.module

    vision_language_encoder.eval()
    tokenizer = utils.get_tokenizer(args.text_encoder)

    x1 = []
    x2 = []
    for _, batch in enumerate(loader):
        with torch.no_grad():
            batch['captions'] = utils.tokenize(batch['captions'], tokenizer, args.text_encoder)
            image_embeddings, text_embeddings = forward_pass_method(vision_language_encoder, batch)
            x1.append(image_embeddings)
            x2.append(text_embeddings)

    x1 = torch.cat(x1, dim=0)
    x2 = torch.cat(x2, dim=0)

    x1 = F.normalize(x1, dim = 1)
    x2 = F.normalize(x2, dim = 1)

    sim_1_2 = torch.matmul(x1, x2.T)
    utils.get_retrieval_score(sim_1_2, log_name='v_t', accelerator=accelerator)
    sim_2_1 = sim_1_2.T
    utils.get_retrieval_score(sim_2_1, log_name='t_v', accelerator=accelerator)

if __name__ == '__main__':
    main()