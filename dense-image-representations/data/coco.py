import re
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import numpy as np
from .datautils import get_image_tokens, get_return_captions

class CocoImagesAndCaptions(CocoDetection):
    def __init__(self, root, annFile, transform, caption_return_policy='all'):
        super().__init__(root, annFile, transform)
        self.annFile = annFile
        self.caption_return_policy = caption_return_policy
        
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        captions = [t['caption'] for t in target]
        captions = get_return_captions(captions, self.caption_return_policy)

        im_size = np.array(image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            if isinstance(image, dict):
                image = image['pixel_values'][0]

        ret = {
            'images': image,
            'im_sizes': im_size,
            'ids': id,
            'captions': captions
        }
        return ret


class CocoImageTokensAndCaptions(CocoDetection):
    def __init__(self, root, image_tokens_root, annFile, transform=None, caption_return_policy='all'):
        super().__init__(root, annFile, transform)
        self.annFile = annFile
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy
        
    def __getitem__(self, index):
        id = self.ids[index]
        all_tokens_dict = get_image_tokens(self.image_tokens_root, id, 0)
        target = self._load_target(id)
        captions = [t['caption'] for t in target]
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict