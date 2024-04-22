import argparse 
import glob
import os

import pdb

from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.tokens import VisualAndTextTokens
from modules import VisionLanguageEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--vision_tokens', type=str, required=True)
parser.add_argument('--text_tokens', type=str, required=True)

parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--projection_dim', type=int, default=128)


args = parser.parse_args()

dataset = VisualAndTextTokens(image_root=args.vision_tokens, text_root=args.text_tokens, number_of_images_per_text=1, random_sample_text=False)

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
# if len(ckpts) == 0:
#     exit(f"No checkpoints found in results/{args.exp_name}")

# print(f"Loading state dict {ckpts[0]}")
# state = torch.load(ckpts[0])
# vision_language_encoder.load_state_dict(state['state_dict'])

correct = 0
total_count = 0
for _, batch in enumerate(loader):
    image_tokens = batch[0].cuda()
    image_attention_mask = batch[1].cuda()
    text_tokens = batch[2].cuda()

    image_output, text_output = vision_language_encoder(image_tokens, image_attention_mask, torch.stack(text_tokens))

    image_embeddings = F.normalize(image_output, dim=-1)
    text_embeddings = F.normalize(text_output, dim=-1)

    num_captions_per_image = text_embeddings.shape[0] // image_embeddings.shape[0]

    for i in range(image_embeddings.shape[0]):
        image_embedding = image_embeddings[i]
        caption_embeddings = text_embeddings[i*num_captions_per_image:(i+1)*num_captions_per_image]

        scores = caption_embeddings @ image_embedding
        correct += (scores.argmax() == 0).int().item()
        total_count += 1

print(f"Accuracy: {correct / total_count}")


    


    