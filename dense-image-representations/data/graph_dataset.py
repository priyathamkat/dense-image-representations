import os
import random
import re
import glob
import json
import torch
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    """"Creates a graph dataset from saved graph data '.pt' files in the root folder."""
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        files = glob.glob(f'{self.root}/*.pt')
        return files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        if self.transform is not None:
            data = self.transform(data)
        return data


class ImageTextGraphDataset(Dataset):
    """"Creates a graph dataset from saved image and text graph data '.pt' files in the root folders."""
    def __init__(self, image_root, text_root, transform=None, pre_transform=None, pre_filter=None):
        self.image_root = image_root
        self.text_root = text_root
        self.transform = transform
        super().__init__(image_root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        saved_ids = list(set([re.findall(r'(\d+)_\d+', i)[0] for i in glob.glob(f'{self.text_root}/*.pt')]))
        return saved_ids

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        image_data = torch.load(f'{self.image_root}/{self.processed_file_names[idx]}_0.pt', map_location = 'cpu')
        text_graphs = glob.glob(f'{self.text_root}/{self.processed_file_names[idx]}_*.pt')
        text_data = torch.load(random.choice(text_graphs), map_location = 'cpu')
        image_data.edge_index = image_data.edge_index.to(torch.long)
        text_data.edge_index = text_data.edge_index.to(torch.long)

        print(image_data)
        if self.transform is not None:
            image_data = self.transform(image_data)
            text_data = self.transform(text_data)
        return image_data, text_data


class ImageGraphTextCaptionDataset(Dataset):
    """"Creates a graph dataset from saved image graph '.pt' files and text caption tokens in the root folders."""
    def __init__(self, image_root, text_root, tokenizer, transform=None, pre_transform=None, pre_filter=None):
        self.image_root = image_root
        self.text_root = text_root
        self.tokenizer = tokenizer
        self.transform = transform
        super().__init__(image_root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        saved_ids = list(set([re.findall(r'(\d+)_\d+', i)[0] for i in glob.glob(f'{self.text_root}/*.json')]))
        return saved_ids

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        image_data = torch.load(f'{self.image_root}/{self.processed_file_names[idx]}_0.pt', map_location = 'cpu')
        text_captions = glob.glob(f'{self.text_root}/{self.processed_file_names[idx]}_*.json')
        with open(random.choice(text_captions), 'r') as file:
            text_caption = json.load(file)['caption']
        text_data = self.tokenizer(text_caption, return_tensors="pt", padding=True).input_ids
        image_data.text_tokens = text_data
        image_data.edge_index = image_data.edge_index.to(torch.long)
        
        print(image_data)
        if self.transform is not None:
            image_data = self.transform(image_data)

        return image_data
