import glob
import re
import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn import ConstantPad2d

import pdb

class VisualAndTextTokens(Dataset):
    def __init__(self, image_root, text_root, image_transform=None, text_transform=None):
        super(VisualAndTextTokens).__init__()
        self.image_root = image_root
        self.text_root = text_root
        self.image_transform = image_transform
        self.text_transform = text_transform

        saved_ids = [re.findall(r'(\d+)_(\d+)', i)[0] for i in glob.glob(f'{self.text_root}/*.pt')]
        saved_ids_dict = {}
        for i in saved_ids:
            if i[0] not in saved_ids_dict:
                saved_ids_dict[int(i[0])] = []
            saved_ids_dict[int(i[0])].append(int(i[1]))

        self.saved_ids = list(saved_ids_dict.keys())
        self.saved_ids_dict = saved_ids_dict

    def __len__(self):
        return len(self.saved_ids)

    def __getitem__(self, idx):
        image_tokens = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_0_tokens.pt', map_location = 'cpu')
        image_attention_mask = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_0_attention_mask.pt', map_location = 'cpu')

        pad = 77 - image_tokens.shape[0]
        image_tokens = ConstantPad2d((0, 0, 0, pad), 0)(image_tokens)
        image_attention_mask = ConstantPad2d((0, pad, 0, pad), 0)(image_attention_mask)

        text_tokens = torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{random.choice(self.saved_ids_dict[self.saved_ids[idx]])}_tokens.pt', map_location = 'cpu')

        if self.image_transform is not None:
            image_tokens = self.image_transform(image_tokens)
            image_attention_mask = self.image_transform(image_attention_mask)
        if self.text_transform is not None:
            text_tokens = self.text_transform(text_tokens)

        return image_tokens, image_attention_mask, text_tokens

    