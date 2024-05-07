import argparse 
import glob
import os

import pdb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip

from transformers import ViTImageProcessor

from data.tokens import VisualAndTextTokens
from data.aro import AROImagesAndCaptionTokens
from modules import VisionLanguageEncoder, VisionLanguageEncoderBase
from contrastive_train import forward_pass
from contrastive_train_baseline import forward_pass as forward_pass_base

TEXTS_PER_IMAGE = {
    'aro_vgr': 2, 
    'aro_vga': 2,
    'aro_coco_order': 5
}

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--dataset', type=str, required=True)

parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--projection_dim', type=int, default=128)
parser.add_argument('--preembed_nodes', action='store_true')

parser.add_argument('--text_encoder', type=str, default='t5')
parser.add_argument('--image_encoder', type=str, default='vit')


args = parser.parse_args()

clip_model = None
if 'clip' in [args.image_encoder, args.text_encoder]:
    clip_model, clip_image_processor = clip.load("ViT-B/16", device='cuda')
    clip_model = clip_model.to(torch.float32)

if 'baseline' in args.exp_name:
    vision_language_encoder = VisionLanguageEncoderBase(projection_dim=args.projection_dim,
                                                        text_encoder=args.text_encoder,
                                                        image_encoder=args.image_encoder,
                                                        clip_model=clip_model,)

    if args.image_encoder == 'vit':
        image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        image_processor = clip_image_processor

    dataset = AROImagesAndCaptionTokens(root='/cmlscratch/nehamk/datasets/aro',
                                        text_tokens_root=f"{args.dataset}_text_tokens",
                                        vit_processor=image_processor,
                                        task=args.dataset,
                                        text_tokenizer_type=args.text_encoder)
    forward_pass = forward_pass_base

else:
    dataset = VisualAndTextTokens(image_root=f"{args.dataset}_visual_tokens",
                                    text_root=f"{args.dataset}_text_tokens", 
                                    number_of_images_per_text=1,
                                    random_sample_text=False,
                                    text_tokenizer_type=args.text_encoder)

    vision_language_encoder = VisionLanguageEncoder(projection_dim=args.projection_dim,
                                                    transformer_width=512, 
                                                    transformer_heads=args.num_heads, 
                                                    transformer_layers=args.num_layers,
                                                    image_embedding_size=2880,
                                                    preembed_nodes=args.preembed_nodes,
                                                    text_encoder=args.text_encoder,
                                                    clip_model=clip_model,)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

vision_language_encoder = vision_language_encoder.cuda()
if 'baseline' in args.exp_name:
    vision_language_encoder = nn.DataParallel(vision_language_encoder)

# ckpts = sorted(glob.glob(f'results/{args.exp_name}/model_*.pth.tar'), key=os.path.getmtime, reverse=True)
# # ckpts = sorted(glob.glob(f'results/{args.exp_name}/model_*.pt'), key=os.path.getmtime, reverse=True)
# if len(ckpts) == 0:
#     exit(f"No checkpoints found in results/{args.exp_name}")

# print(f"Loading state dict {ckpts[0]}")
# state = torch.load(ckpts[0])
# vision_language_encoder.load_state_dict(state['state_dict'])
# # vision_language_encoder.load_state_dict(state)
vision_language_encoder.eval()

correct = 0
total_count = 0
for _, batch in enumerate(loader):
    batch[-1] = torch.cat(batch[-1]) # multiple captions per image
    with torch.no_grad():
        image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    num_captions_per_image = text_embeddings.shape[0] // image_embeddings.shape[0]

    for i in range(image_embeddings.shape[0]):
        image_embedding = image_embeddings[i]
        caption_embeddings = torch.cat([text_embeddings[i + j * image_embeddings.shape[0]].unsqueeze(0) for j in range(num_captions_per_image)])
        scores = caption_embeddings @ image_embedding
        if 'coco_order' not in args.dataset:
            correct += (scores.argmax() == 1).int().item()
        else:    
            correct += (scores.argmax() == 0).int().item()
        total_count += 1

print(f"Accuracy: {correct / total_count}")


    


    