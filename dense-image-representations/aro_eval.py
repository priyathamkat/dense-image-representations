import argparse 
import glob
import os
import pandas as pd
import numpy as np

import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import clip
from transformers import ViTImageProcessor

from data.datautils import get_dataset
import utils
from modules import VisionLanguageEncoder, VisionLanguageEncoderBase

from accelerate.utils import DistributedDataParallelKwargs
from accelerate import Accelerator

@torch.no_grad()
def get_retrieval_scores_batched(model, tokenizer, joint_loader, args):
    """Computes the scores for each image_option / caption_option pair in the joint loader.

    Args:
        joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
        "image_options" is a list of images, and "caption_options" is a list of captions.

    Returns:
        all_scores: A numpy array containing the scores of the shape NxKxL,
        where N is the number of test cases, K is the number of image options per the test case,
        and L is the number of caption options per the test case.
    """
    scores = []
    for _, batch in enumerate(joint_loader):
        if 'baseline' not in args.exp_name:
            image_embeddings = model.encode_image(batch['image_tokens'],
                                                    batch['image_features'],
                                                    batch['num_non_pad_tokens'],
                                                    batch['num_nodes'],
                                                    batch['image_attention_mask'])
            image_embeddings = image_embeddings.mean(dim=1).cpu().numpy()
        else:
            image_embeddings = model.encode_image(batch['images']).cpu().numpy() # B x D

        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
        image_options = np.expand_dims(image_embeddings, axis=1) # B x K x D
        
        caption_options = []
        for c_option in batch["captions"]:
            caption_tokenized = utils.tokenize(c_option, tokenizer, args.text_encoder).view(-1, 77)
            caption_embeddings = model.encode_text(caption_tokenized.cuda()).cpu().numpy() # B x D
            caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D
            caption_options.append(np.expand_dims(caption_embeddings, axis=1))
        
        caption_options = np.concatenate(caption_options, axis=1) # B x L x D
        batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
        scores.append(batch_scores)
    
    all_scores = np.concatenate(scores, axis=0) # N x K x L
    return all_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--dataset', type=str, required=True)

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
            dataset_name = args.dataset,
            transform = image_processor,
            with_image_tokens = False, 
            caption_return_policy = 'all',
            hf_vit_processor = 'vit' in args.image_encoder,
        )

    else:
        dataset = get_dataset(
            dataset_name = args.dataset,
            image_tokens_root = f'{args.dataset}_visual_tokens',
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

    scores = get_retrieval_scores_batched(model = vision_language_encoder,
                                            tokenizer = tokenizer,
                                            joint_loader = loader,
                                            args=args)

    records = dataset.evaluate_scores(scores)

    if 'vgr' in args.dataset:
        vgr_records = records
        symmetric = ['adjusting', 'attached to', 'between', 'bigger than', 'biting', 'boarding', 'brushing', 'chewing', 'cleaning', 'climbing', 'close to', 'coming from', 'coming out of', 'contain', 'crossing', 'dragging', 'draped over', 'drinking', 'drinking from', 'driving', 'driving down', 'driving on', 'eating from', 'eating in', 'enclosing', 'exiting', 'facing', 'filled with', 'floating in', 'floating on', 'flying', 'flying above', 'flying in', 'flying over', 'flying through', 'full of', 'going down', 'going into', 'going through', 'grazing in', 'growing in', 'growing on', 'guiding', 'hanging from', 'hanging in', 'hanging off', 'hanging over', 'higher than', 'holding onto', 'hugging', 'in between', 'jumping off', 'jumping on', 'jumping over', 'kept in', 'larger than', 'leading', 'leaning over', 'leaving', 'licking', 'longer than', 'looking in', 'looking into', 'looking out', 'looking over', 'looking through', 'lying next to', 'lying on top of', 'making', 'mixed with', 'mounted on', 'moving', 'on the back of', 'on the edge of', 'on the front of', 'on the other side of', 'opening', 'painted on', 'parked at', 'parked beside', 'parked by', 'parked in', 'parked in front of', 'parked near', 'parked next to', 'perched on', 'petting', 'piled on', 'playing', 'playing in', 'playing on', 'playing with', 'pouring', 'reaching for', 'reading', 'reflected on', 'riding on', 'running in', 'running on', 'running through', 'seen through', 'sitting behind', 'sitting beside', 'sitting by', 'sitting in front of', 'sitting near', 'sitting next to', 'sitting under', 'skiing down', 'skiing on', 'sleeping in', 'sleeping on', 'smiling at', 'sniffing', 'splashing', 'sprinkled on', 'stacked on', 'standing against', 'standing around', 'standing behind', 'standing beside', 'standing in front of', 'standing near', 'standing next to', 'staring at', 'stuck in', 'surrounding', 'swimming in', 'swinging', 'talking to', 'topped with', 'touching', 'traveling down', 'traveling on', 'tying', 'typing on', 'underneath', 'wading in', 'waiting for', 'walking across', 'walking by', 'walking down', 'walking next to', 'walking through', 'working in', 'working on', 'worn on', 'wrapped around', 'wrapped in', 'by', 'of', 'near', 'next to', 'with', 'beside', 'on the side of', 'around']
        df = pd.DataFrame(vgr_records)
        df = df[~df.Relation.isin(symmetric)]
        print(f"Accuracy: {df.Accuracy.mean()}")

    elif 'vga' in args.dataset:
        vga_records = records
        df = pd.DataFrame(vga_records)
        print(f"Accuracy: {df.Accuracy.mean()}")

    else:
        coco_order_records = records
        df = pd.DataFrame(coco_order_records)
        print(f"Accuracy: {df['Precision@1'].mean()}")


if __name__ == '__main__':
    main()