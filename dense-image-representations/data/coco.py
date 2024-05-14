import re
import glob
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import numpy as np
from .datautils import get_image_tokens, get_return_captions

import pdb

class CocoImagesAndCaptions(CocoDetection):
    def __init__(self, root, annFile, transform, caption_return_policy='all', hf_vit_processor = False):
        super().__init__(root, annFile, transform)
        self.annFile = annFile
        self.caption_return_policy = caption_return_policy
        self.transform = transform
        self.hf_vit_processor = hf_vit_processor
        
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)
        captions = [t['caption'] for t in target]
        captions = get_return_captions(captions, self.caption_return_policy)

        im_size = np.array(image.size)

        if self.transform is not None:
            image = self.transform(image)
            if self.hf_vit_processor:
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

        existing_ids = [int(re.findall(r'(\d+)_0', f)[0]) for f in glob.glob(f'{image_tokens_root}/*_0_edge_tokens.pt')]
        self.ids = [i for i in self.ids if i in existing_ids]
        
    def __getitem__(self, index):
        id = self.ids[index]
        all_tokens_dict = get_image_tokens(self.image_tokens_root, id, 0)
        target = self._load_target(id)
        captions = [t['caption'] for t in target]
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict