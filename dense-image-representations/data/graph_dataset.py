import os
import random
import re
import glob
import torch
from torch_geometric.data import Dataset

class GraphDataset(Dataset):
    """"Creates a graph dataset from saved graph data '.pt' files in the root folder."""
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        files = glob.glob(f'{self.root}/*.pt')
        return files

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        return data


class ImageTextGraphDataset(Dataset):
    """"Creates a graph dataset from saved image and text graph data '.pt' files in the root folders."""
    def __init__(self, image_root, text_root, transform=None, pre_transform=None, pre_filter=None):
        self.image_root = image_root
        self.text_root = text_root
        super().__init__(image_root, transform, pre_transform, pre_filter)

    @property
    def processed_file_names(self):
        saved_ids = list(set([re.findall(r'(\d+)_\d+', i)[0] for i in glob.glob(f'{self.text_root}/*.pt')]))
        return saved_ids

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        image_data = torch.load(f'{self.image_root}/{self.processed_file_names[idx]}_0.pt')
        text_graphs = glob.glob(f'{self.text_root}/{self.processed_file_names[idx]}_*.pt')
        text_data = torch.load(random.choice(text_graphs))
        image_data.edge_index = image_data.edge_index.to(torch.long)
        text_data.edge_index = text_data.edge_index.to(torch.long)
        return image_data, text_data