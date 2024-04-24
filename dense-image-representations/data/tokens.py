import glob
import os
import re
import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn import ConstantPad2d

import pdb

class VisualAndTextTokens(Dataset):
    def __init__(self, image_root, text_root, image_transform=None, text_transform=None, number_of_images_per_text=1, number_of_texts_per_image=5, random_sample_text=True):
        super(VisualAndTextTokens).__init__()
        self.image_root = image_root
        self.text_root = text_root
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.number_of_images_per_text = number_of_images_per_text
        self.number_of_texts_per_image = number_of_texts_per_image
        self.random_sample_text = random_sample_text

        saved_ids = [re.findall(r'(\d+)_(\d+)', i)[0] for i in glob.glob(f'{self.text_root}/*.pt')]
        saved_ids_dict = {}
        for i in saved_ids:
            if i[0] not in saved_ids_dict:
                saved_ids_dict[int(i[0])] = []
            saved_ids_dict[int(i[0])].append(int(i[1]))

        self.saved_ids = list(saved_ids_dict.keys())
        self.saved_ids_dict = saved_ids_dict

        self.image_id_to_num_node_tokens = None
        if os.path.exists(f'{image_root}/image_id_to_num_node_tokens.json'):
            with open(f'{image_root}/image_id_to_num_node_tokens.json', 'r') as f:
                self.image_id_to_num_node_tokens = json.load(f)

    def __len__(self):
        return len(self.saved_ids)

    def __getitem__(self, idx):
        if self.number_of_images_per_text == 1:
            image_tokens = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_0_tokens.pt', map_location = 'cpu')
            image_attention_mask = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_0_attention_mask.pt', map_location = 'cpu')
            pad = 77 - image_tokens.shape[0]
            image_tokens = ConstantPad2d((0, 0, 0, pad), 0)(image_tokens)
            image_attention_mask = ConstantPad2d((0, pad, 0, pad), 0)(image_attention_mask)

            if self.random_sample_text:
                text_tokens = torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{random.choice(self.saved_ids_dict[self.saved_ids[idx]])}_tokens.pt', map_location = 'cpu')
            else:
                text_tokens_list = []
                for i in range(self.number_of_texts_per_image):
                    text_tokens_list.append(torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{i}_tokens.pt', map_location = 'cpu'))
                text_tokens = text_tokens_list

        else:
            image_tokens_list = []
            image_attention_mask_list = []
            text_tokens_list = []
            for i in range(self.number_of_images_per_text):
                image_tokens = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_tokens.pt', map_location = 'cpu')
                image_attention_mask = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_attention_mask.pt', map_location = 'cpu')
                pad = 77 - image_tokens.shape[0]
                image_tokens = ConstantPad2d((0, 0, 0, pad), 0)(image_tokens)
                image_attention_mask = ConstantPad2d((0, pad, 0, pad), 0)(image_attention_mask)

                text_tokens = torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{i}_tokens.pt', map_location = 'cpu')

                if self.image_transform is not None:
                    image_tokens = self.image_transform(image_tokens)
                    image_attention_mask = self.image_transform(image_attention_mask)
                if self.text_transform is not None:
                    text_tokens = self.text_transform(text_tokens)

                image_tokens_list.append(image_tokens)
                image_attention_mask_list.append(image_attention_mask)
                text_tokens_list.append(text_tokens)

            image_tokens = image_tokens_list
            image_attention_mask = image_attention_mask_list
            text_tokens = text_tokens_list

        if self.image_id_to_num_node_tokens is not None:
            return image_tokens, image_attention_mask, text_tokens, self.image_id_to_num_node_tokens[self.saved_ids[idx]]
        
        return image_tokens, image_attention_mask, text_tokens

    
