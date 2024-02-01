import os
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