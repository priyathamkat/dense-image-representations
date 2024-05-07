from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
import torch

text_tokenizer_type_str = {
    't5_small': '',
    't5_base': '_t5_base',
    'clip': '_clip'
}

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


class WinogroundImagesAndCaptionTokens(Dataset):
    def __init__(self, root, text_tokens_root, vit_processor=None, text_tokenizer_type='t5_small'):
        winoground_dataset = load_dataset('facebook/winoground', token='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', cache_dir = root, trust_remote_code = True)
        self.data = winoground_dataset['test']
        self.vit_processor = vit_processor
        self.text_tokens_root = text_tokens_root
        self.text_tokenizer_str = text_tokenizer_type_str[text_tokenizer_type]

    def __len__(self):
        return len(self.data)
        # return 10

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        im0 = sample_dict['image_0'].convert('RGB')
        im1 = sample_dict['image_1'].convert('RGB')

        text_tokens_0 = torch.load(f'{self.text_tokens_root}/{idx}_0{self.text_tokenizer_str}_tokens.pt', map_location='cpu')
        text_tokens_1 = torch.load(f'{self.text_tokens_root}/{idx}_1{self.text_tokenizer_str}_tokens.pt', map_location='cpu')

        if self.vit_processor:

            im0 = self.vit_processor(im0)
            im1 = self.vit_processor(im1)
            if not isinstance(im0, torch.Tensor):
                im0 = im0['pixel_values'][0]
                im1 = im1['pixel_values'][0]

        return [im0, im1], [text_tokens_0, text_tokens_1]
            