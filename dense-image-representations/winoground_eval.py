import argparse 
import glob
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip
from transformers import ViTImageProcessor

from modules import VisionLanguageEncoder, VisionLanguageEncoderBase

from data.datautils import get_dataset
import utils

from contrastive_train import forward_pass
from contrastive_train_baseline import forward_pass as forward_pass_base

from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator

import pdb

def text_correct(result, index):
    return result["c0_i0"][index] > result["c1_i0"][index] and result["c1_i1"][index] > result["c0_i1"][index]

def image_correct(result, index):
    return result["c0_i0"][index] > result["c0_i1"][index] and result["c1_i1"][index] > result["c1_i0"][index]

def group_correct(result, index):
    return image_correct(result, index) and text_correct(result, index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--vision_tokens', type=str, default='winoground_visual_tokens')
    parser.add_argument('--text_tokens', type=str, default='winoground_text_tokens')

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
        clip_model, clip_image_processor = clip.load("ViT-B/32")
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
            dataset_name = 'winoground',
            transform = image_processor,
            with_image_tokens = False, 
            caption_return_policy = 'all',
            hf_vit_processor = 'vit' in args.image_encoder,
        )
        forward_pass_method = forward_pass_base

    else:
        dataset = get_dataset(
            dataset_name = 'winoground',
            image_tokens_root = f'winoground_visual_tokens',
            with_image_tokens = True, 
            caption_return_policy = 'all'
        )

        vision_language_encoder = VisionLanguageEncoder(projection_dim=args.projection_dim,
                                                        transformer_width=768 if args.transformer == 'clip' else 512, 
                                                        transformer_heads=args.num_heads, 
                                                        transformer_layers=args.num_layers,
                                                        image_embedding_size=2880,
                                                        preembed_nodes='_preembed' in args.exp_name,
                                                        text_encoder=args.text_encoder,
                                                        clip_model=clip_model,
                                                        transformer=args.transformer,
                                                        use_attention_mask=args.use_attention_mask,
                                                        )

        forward_pass_method = forward_pass

    tokenizer = utils.get_tokenizer(args.text_encoder)

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

    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0

    total_count = 0
    for _, batch in enumerate(loader):
        with torch.no_grad():
            batch_0 = {k:v[0] for (k,v) in zip(batch.keys(), batch.values())}
            batch_1 = {k:v[1] for (k,v) in zip(batch.keys(), batch.values())}
            batch_0['image_attention_mask'] = None
            batch_1['image_attention_mask'] = None

            batch_0['captions'] = utils.tokenize(batch_0['captions'], tokenizer, args.text_encoder)
            batch_1['captions'] = utils.tokenize(batch_1['captions'], tokenizer, args.text_encoder)
            image_embeddings_0, text_embeddings_0 = forward_pass_method(vision_language_encoder, batch_0)
            image_embeddings_1, text_embeddings_1 = forward_pass_method(vision_language_encoder, batch_1)

            image_embeddings_0 = F.normalize(image_embeddings_0, dim=-1)
            image_embeddings_1 = F.normalize(image_embeddings_1, dim=-1)
            text_embeddings_0 = F.normalize(text_embeddings_0, dim=-1)
            text_embeddings_1 = F.normalize(text_embeddings_1, dim=-1)

            sim_c0_i0 = (text_embeddings_0 * image_embeddings_0).sum(dim=-1)
            sim_c0_i1 = (text_embeddings_0 * image_embeddings_1).sum(dim=-1)
            sim_c1_i0 = (text_embeddings_1 * image_embeddings_0).sum(dim=-1)
            sim_c1_i1 = (text_embeddings_1 * image_embeddings_1).sum(dim=-1)

            results = {
                'c0_i0': sim_c0_i0,
                'c0_i1': sim_c0_i1,
                'c1_i0': sim_c1_i0,
                'c1_i1': sim_c1_i1,
            }

            for i in range(sim_c0_i0.shape[0]):
                if text_correct(results, i):
                    text_correct_count += 1
                if image_correct(results, i):
                    image_correct_count += 1
                if group_correct(results, i):
                    group_correct_count += 1

            total_count += sim_c0_i0.shape[0]

    print(f"Text Correct: {text_correct_count * 100 / total_count}")
    print(f"Image Correct: {image_correct_count * 100 / total_count}")
    print(f"Group Correct: {group_correct_count * 100 / total_count}")


if __name__ == "__main__":
    main()