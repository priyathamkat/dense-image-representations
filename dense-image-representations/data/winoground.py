from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np

class Winoground(Dataset):
    def __init__(self, root, transform=None):
        winoground_dataset = load_dataset('facebook/winoground', token='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', cache_dir = root, trust_remote_code = True)
        self.data = winoground_dataset['test']
        self.transform = transform

    def __len__(self):
        return len(self.data)
        # return 10

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        im0 = sample_dict['image_0']
        im1 = sample_dict['image_1']

        im0_size = np.array(im0.size)
        im1_size = np.array(im1.size)
        # cap0 = sample_dict['caption_0']
        # cap1 = sample_dict['caption_1']
        if self.transform:
            im0 = self.transform(im0)
            im1 = self.transform(im1)

        return [im0, im1], [im0_size, im1_size], [idx, idx]
            