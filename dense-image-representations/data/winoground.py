from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from .datautils import get_image_tokens, get_return_captions

class WinogroundImagesAndCaptions(Dataset):
    def __init__(self, root, transform=None, hf_vit_processor = False, caption_return_policy='all'):
        winoground_dataset = load_dataset('facebook/winoground', token='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', cache_dir = root, trust_remote_code = True)
        self.data = winoground_dataset['test']
        self.transform = transform
        self.hf_vit_processor = hf_vit_processor
        self.caption_return_policy=caption_return_policy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        im0 = sample_dict['image_0'].convert('RGB')
        im1 = sample_dict['image_1'].convert('RGB')

        im0_size = np.array(im0.size)
        im1_size = np.array(im1.size)
        cap0 = sample_dict['caption_0']
        cap1 = sample_dict['caption_1']

        if self.transform:
            im0 = self.transform(im0)
            im1 = self.transform(im1)
            if self.hf_vit_processor:
                im0 = im0['pixel_values'][0]
                im1 = im1['pixel_values'][0]

        ret = {
            'images': [im0, im1],
            'im_sizes': [im0_size, im1_size],
            'ids': idx,
            'captions': [cap0, cap1]
        }
        return ret


class WinogroundImageTokensAndCaptions(Dataset):
    def __init__(self, root, image_tokens_root, transform=None, caption_return_policy='all'):
        winoground_dataset = load_dataset('facebook/winoground', token='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', cache_dir = root, trust_remote_code = True)
        self.data = winoground_dataset['test']
        self.transform = transform
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        all_tokens_dict_0 = get_image_tokens(self.image_tokens_root, idx, 0)
        all_tokens_dict_1 = get_image_tokens(self.image_tokens_root, idx, 1)

        all_tokens = {}
        for k in all_tokens_dict_0:
            all_tokens[k] = [all_tokens_dict_0[k], all_tokens_dict_1[k]]

        all_tokens['captions'] = [sample_dict['caption_0'], sample_dict['caption_1']]

        return all_tokens
