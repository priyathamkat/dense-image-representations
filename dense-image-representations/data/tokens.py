import glob
import os
import re
import json
import random
import torch
from torch.utils.data import Dataset
from torch.nn import ConstantPad2d

import pdb

text_tokenizer_type_str = {
    't5_small': '',
    't5_base': '_t5_base',
    'clip': '_clip'
}

class VisualAndTextTokens(Dataset):
    def __init__(self, 
                 image_root, 
                 text_root, 
                 image_transform=None, 
                 text_transform=None, 
                 number_of_images_per_text=1,
                 text_tokenizer_type='t5_small',
                 random_sample_text=True):
        super(VisualAndTextTokens).__init__()
        self.image_root = image_root
        self.text_root = text_root
        self.image_transform = image_transform
        self.text_transform = text_transform
        self.number_of_images_per_text = number_of_images_per_text
        self.random_sample_text = random_sample_text

        self.text_tokenizer_str = text_tokenizer_type_str[text_tokenizer_type]
        saved_ids = [re.findall(r'(\d+)_(\d+)', i)[0] for i in sorted(glob.glob(f'{self.text_root}/*{self.text_tokenizer_str}_tokens.pt'))]
        # existing_ids = [re.findall(r'(\d+)_\d+', i)[0] for i in glob.glob(f'{self.image_root}/*_edge_tokens.pt')]
        # saved_ids = [i for i in saved_ids if i[0] in existing_ids]
        saved_ids_dict = {}
        for i in saved_ids:
            if int(i[0]) not in saved_ids_dict:
                saved_ids_dict[int(i[0])] = []
            saved_ids_dict[int(i[0])].append(int(i[1]))

        self.saved_ids = list(saved_ids_dict.keys())
        self.saved_ids_dict = saved_ids_dict

    def __len__(self):
        return len(self.saved_ids)

    def get(self, idx, i):
        node_tokens = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_node_tokens.pt', map_location = 'cpu')
        edge_tokens = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_edge_tokens.pt', map_location = 'cpu')
        image_features = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_image_features.pt', map_location = 'cpu')
        num_non_pad_tokens = node_tokens.shape[0] + edge_tokens.shape[0]
        num_nodes = node_tokens.shape[0]
        image_tokens = torch.cat([node_tokens, edge_tokens], dim=0)
        
        image_attention_mask = torch.load(f'{self.image_root}/{self.saved_ids[idx]}_{i}_attention_mask.pt', map_location = 'cpu')
        pad = 77 - image_tokens.shape[0]
        image_tokens = ConstantPad2d((0, 0, 0, pad), 0)(image_tokens)
        image_attention_mask = ConstantPad2d((0, pad, 0, pad), 0)(image_attention_mask)

        return image_tokens, image_features, num_non_pad_tokens, num_nodes, image_attention_mask

    def __getitem__(self, idx):
        if self.number_of_images_per_text == 1:
            image_tokens, image_features, num_non_pad_tokens, num_nodes, image_attention_mask = self.get(idx, 0)
            
            if self.random_sample_text:
                text_tokens = torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{random.choice(self.saved_ids_dict[self.saved_ids[idx]])}{self.text_tokenizer_str}_tokens.pt', map_location = 'cpu')
            else:
                text_tokens_list = []
                for text_id in self.saved_ids_dict[self.saved_ids[idx]]:
                    text_tokens_list.append(torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{text_id}{self.text_tokenizer_str}_tokens.pt', map_location = 'cpu'))
                text_tokens = text_tokens_list

        else:
            image_tokens_list = [] 
            image_features_list = [] 
            num_non_pad_tokens_list = [] 
            num_nodes_list = []
            image_attention_mask_list = [] 
            text_tokens_list = []
            for i in range(self.number_of_images_per_text):
                image_tokens, image_features, num_non_pad_tokens, num_nodes, image_attention_mask = self.get(idx, i)

                text_tokens = torch.load(f'{self.text_root}/{self.saved_ids[idx]}_{self.saved_ids_dict[self.saved_ids[idx]][i]}{self.text_tokenizer_str}_tokens.pt', map_location = 'cpu')

                image_tokens_list.append(image_tokens)
                image_features_list.append(image_features)
                num_non_pad_tokens_list.append(num_non_pad_tokens)
                num_nodes_list.append(num_nodes)
                image_attention_mask_list.append(image_attention_mask)
                text_tokens_list.append(text_tokens)

            image_tokens = image_tokens_list
            image_features = image_features_list
            num_non_pad_tokens = num_non_pad_tokens_list
            num_nodes = num_nodes_list
            image_attention_mask = image_attention_mask_list
            text_tokens = text_tokens_list
        
        return image_tokens, image_features, num_non_pad_tokens, num_nodes, image_attention_mask, text_tokens

    
