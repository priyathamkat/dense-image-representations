import re
import glob
from torch.utils.data import Dataset
import numpy as np
from .datautils import get_image_tokens, get_return_captions

import pandas as pd 

import pdb

class FlickrImagesAndCaptions(Dataset):
    def __init__(self, root, caption_file, transform, caption_return_policy='all', hf_vit_processor = False):
        self.root = root
        self.caption_return_policy = caption_return_policy
        self.transform = transform
        self.hf_vit_processor = hf_vit_processor
        self.captiondf = pd.read_csv(caption_file, sep='| ')
        self.image_ids = list(set(self.captiondf['image_name']))
        
    def __getitem__(self, index):
        path = f'{self.root}/{self.image_ids[index]}'

        image = PIL.Image.open(path).convert('RGB')

        captions = self.captiondf[self.captiondf['image_name'] == self.image_ids[index]]['caption'].tolist()
        captions = get_return_captions(captions, self.caption_return_policy)

        im_size = np.array(image.size)

        if self.img_transform is not None:
            image = self.img_transform(image)
            if self.hf_vit_processor:
                image = image['pixel_values'][0]

        ret = {
            'images': image,
            'im_sizes': im_size,
            'ids': id,
            'captions': captions
        }
        return ret


class FlickrImageTokensAndCaptions(Dataset):
    def __init__(self, root, image_tokens_root, annFile, transform=None, caption_return_policy='all'):
        super().__init__(root, annFile, transform)
        self.annFile = annFile
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy
        
    def __getitem__(self, index):
        id = self.ids[index]
        _, target = super().__getitem__(index)
        all_tokens_dict = get_image_tokens(self.image_tokens_root, id, 0)
        captions = [t['caption'] for t in target]
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict