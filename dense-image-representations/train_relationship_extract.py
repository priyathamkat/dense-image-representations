import argparse

import torch
from data_processing.visual_genome import VisualGenomeRelationshipWithSAMAndT5
from segment_anything import sam_model_registry
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, T5EncoderModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_workers", type=int, default=16)

args = parser.parse_args()

device = torch.device('cuda')
sam = sam_model_registry['vit_b'](checkpoint='../sam_vit_b_01ec64.pth')
t5_tokenizer = AutoTokenizer.from_pretrained('t5-small')
t5_model = T5EncoderModel.from_pretrained('t5-small')

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

dataset = VisualGenomeRelationshipWithSAMAndT5(root_dir='/cmlscratch/nehamk/datasets/visual_genome', sam_model=sam, t5_tokenizer=t5_tokenizer, t5_encoder=t5_model, transform=transform)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

sam_image_encoder = sam.image_encoder.to(device)

for i, batch in enumerate(dataloader):
    img, object_bbox_sam_emb, subject_bbox_sam_emb, predicate_emb = batch
    img = img.to(device)
    object_bbox_sam_emb = object_bbox_sam_emb.to(device)
    subject_bbox_sam_emb = subject_bbox_sam_emb.to(device)
    predicate_emb = predicate_emb.to(device)

    # train relationship extractor
    