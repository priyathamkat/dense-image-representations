import argparse 
import glob
import os

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from modules import VisionLanguageEncoder

from data.winoground import Winoground
from data.tokens import VisualAndTextTokens


import pdb

def text_correct(result, index):
    return result["c0_i0"][index] > result["c1_i0"][index] and result["c1_i1"][index] > result["c0_i1"][index]

def image_correct(result, index):
    return result["c0_i0"][index] > result["c0_i1"][index] and result["c1_i1"][index] > result["c1_i0"][index]

def group_correct(result, index):
    return image_correct(result, index) and text_correct(result, index)


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--vision_tokens', type=str, default='winoground_visual_tokens')
parser.add_argument('--text_tokens', type=str, default='winoground_text_tokens')

parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--projection_dim', type=int, default=128)


args = parser.parse_args()

dataset = VisualAndTextTokens(image_root=args.vision_tokens, text_root=args.text_tokens, number_of_images_per_text=2)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)

vision_language_encoder = VisionLanguageEncoder(embed_dim=512,
                                                projection_dim=args.projection_dim, 
                                                transformer_width=512, 
                                                transformer_heads=args.num_heads, 
                                                transformer_layers=args.num_layers)

vision_language_encoder = vision_language_encoder.cuda()

# ckpts = sorted(glob.glob(f'results/{args.exp_name}/model_*.pth.tar'), key=os.path.getmtime, reverse=True)
ckpts = sorted(glob.glob(f'results/{args.exp_name}/model_*.pt'), key=os.path.getmtime, reverse=True)
if len(ckpts) == 0:
    exit(f"No checkpoints found in results/{args.exp_name}")

print(f"Loading state dict {ckpts[0]}")
state = torch.load(ckpts[0])
# vision_language_encoder.load_state_dict(state['state_dict'])
vision_language_encoder.load_state_dict(state)
vision_language_encoder.eval()

text_correct_count = 0
image_correct_count = 0
group_correct_count = 0

total_count = 0
for _, batch in enumerate(loader):
    image_tokens_0 = batch[0][0].cuda()
    image_tokens_1 = batch[0][1].cuda()

    image_attention_mask_0 = batch[1][0].cuda()
    image_attention_mask_1 = batch[1][1].cuda()

    text_tokens_0 = batch[2][0].cuda()
    text_tokens_1 = batch[2][1].cuda()

    with torch.no_grad():
        image_attention_mask_0 = torch.zeros(image_tokens_0.shape[0], image_tokens_0.shape[1], image_tokens_0.shape[1]).to('cuda')
        image_attention_mask_1 = torch.zeros(image_tokens_1.shape[0], image_tokens_1.shape[1], image_tokens_1.shape[1]).to('cuda')

        with torch.no_grad():
            image_embeddings_0, text_embeddings_0 = vision_language_encoder(image_tokens_0, image_attention_mask_0, text_tokens_0)
            image_embeddings_1, text_embeddings_1 = vision_language_encoder(image_tokens_1, image_attention_mask_1, text_tokens_1)

        image_embeddings_0 = image_embeddings_0.mean(dim=1)
        text_embeddings_0 = text_embeddings_0.mean(dim=1)
        image_embeddings_1 = image_embeddings_1.mean(dim=1)
        text_embeddings_1 = text_embeddings_1.mean(dim=1)

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

