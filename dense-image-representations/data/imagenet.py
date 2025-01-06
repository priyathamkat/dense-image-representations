import re
import glob
from torchvision.datasets import ImageFolder
import numpy as np
from .datautils import get_image_tokens, get_return_captions
import json

import pdb

class ImageNetImagesAndCaptions(ImageFolder):
    def __init__(self, root, transform, class_mapping_json, caption_return_policy='all', hf_vit_processor = False):
        super().__init__(root)
        self.caption_return_policy = caption_return_policy
        self.img_transform = transform
        self.hf_vit_processor = hf_vit_processor
        self.class_mapping = json.load(open(class_mapping_json))
        
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        captions = self.class_mapping[str(label)][1]

        im_size = np.array(image.size)

        if self.img_transform is not None:
            image = self.img_transform(image)
            if self.hf_vit_processor:
                image = image['pixel_values'][0]

        ret = {
            'images': image,
            'im_sizes': im_size,
            'ids': idx,
            'captions': captions
        }
        return ret



class ImageNetImageTokensAndCaptions(ImageFolder):
    def __init__(self, root_dir, class_mapping_json, image_tokens_root, caption_return_policy='all'):
        super().__init__(root=root_dir)
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy
        self.class_mapping = json.load(open(class_mapping_json))

    def __getitem__(self, idx):
        _, label = super().__getitem__(idx)
        all_tokens_dict = get_image_tokens(self.image_tokens_root, idx, 0)
        captions = self.class_mapping[str(label)][1]
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict