from torch.utils.data import Dataset
import os
import sys
sys.path.append('/cmlscratch/nehamk/vision-language-models-are-bows')
from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
import torch
import numpy as np
import glob
from PIL import Image
from .datautils import get_image_tokens, get_return_captions


class VG_RelationImagesAndCaptions(VG_Relation):
    def __init__(self,
                root_dir,
                image_preprocess=None,
                download=True,
                return_image_sizes = False,
                caption_return_policy = 'all',
                hf_vit_processor = False):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.return_image_sizes = return_image_sizes
        self.hf_vit_processor = hf_vit_processor
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        image = sample_dict['image_options'][0]
        if self.hf_vit_processor:
            image = image['pixel_values'][0]
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)

        ret = {
            'images': image,
            'ids': idx,
            'captions': captions
        }
        if self.return_image_sizes:
            im_size = np.array(Image.open(self.dataset[idx]['image_path']).convert('RGB').size)
            ret['im_sizes'] = im_size

        return ret

class VG_RelationImageTokensAndCaptions(VG_Relation):
    def __init__(self, root_dir, image_tokens_root, image_preprocess=None, download=True, caption_return_policy='all'):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        all_tokens_dict = get_image_tokens(self.image_tokens_root, idx, 0)
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict


class VG_AttributionImagesAndCaptions(VG_Attribution):
    def __init__(self,
                root_dir,
                image_preprocess=None,
                download=True,
                return_image_sizes = False,
                caption_return_policy = 'all',
                hf_vit_processor = False):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.return_image_sizes = return_image_sizes
        self.hf_vit_processor = hf_vit_processor
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        image = sample_dict['image_options'][0]
        if self.hf_vit_processor:
            image = image['pixel_values'][0]
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)

        ret = {
            'images': image,
            'ids': idx,
            'captions': captions
        }
        if self.return_image_sizes:
            im_size = np.array(Image.open(self.dataset[idx]['image_path']).convert('RGB').size)
            ret['im_sizes'] = im_size

        return ret

class VG_AttributionImageTokensAndCaptions(VG_Attribution):
    def __init__(self, root_dir, image_tokens_root, image_preprocess=None, download=True, caption_return_policy='all'):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        all_tokens_dict = get_image_tokens(self.image_tokens_root, idx, 0)
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict


class COCO_OrderImagesAndCaptions(COCO_Order):
    def __init__(self,
                 root_dir,
                 image_preprocess=None,
                 download=True,
                 return_image_sizes = False,
                 caption_return_policy = 'all',
                 hf_vit_processor = False):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.root_dir = root_dir
        self.return_image_sizes = return_image_sizes
        self.hf_vit_processor = hf_vit_processor
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        image = sample_dict['image_options'][0]
        if self.hf_vit_processor:
            image = image['pixel_values'][0]
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)

        ret = {
            'images': image,
            'ids': idx,
            'captions': captions
        }
        if self.return_image_sizes:
            image_path = os.path.join(self.root_dir, self.test_cases[idx]["image"])   
            im_size = np.array(Image.open(image_path).convert('RGB').size)
            ret['im_sizes'] = im_size

        return ret

class COCO_OrderImageTokensAndCaptions(COCO_Order):
    def __init__(self, root_dir, image_tokens_root, image_preprocess=None, download=True, caption_return_policy='all'):
        super().__init__(image_preprocess=image_preprocess, download=download, root_dir=root_dir)
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        all_tokens_dict = get_image_tokens(self.image_tokens_root, idx, 0)
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict
    

class Flickr_OrderImagesAndCaptions(Flickr30k_Order):
    def __init__(self,
                 root_dir,
                 image_preprocess=None,
                 return_image_sizes = False,
                 caption_return_policy = 'all',
                 hf_vit_processor = False):
        super().__init__(image_preprocess=image_preprocess, split='test', root_dir=root_dir)
        self.return_image_sizes = return_image_sizes
        self.root_dir = root_dir
        self.hf_vit_processor = hf_vit_processor
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        image = sample_dict['image_options'][0]
        if self.hf_vit_processor:
            image = image['pixel_values'][0]
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)

        ret = {
            'images': image,
            'ids': idx,
            'captions': captions
        }
        if self.return_image_sizes:
            image_path = os.path.join(self.root_dir, self.test_cases[idx]["image"])   
            im_size = np.array(Image.open(image_path).convert('RGB').size)
            ret['im_sizes'] = im_size

        return ret

class Flickr_OrderImageTokensAndCaptions(Flickr30k_Order):
    def __init__(self, root_dir, image_tokens_root, image_preprocess=None, caption_return_policy='all'):
        super().__init__(image_preprocess=image_preprocess, split='test', root_dir=root_dir)
        self.image_tokens_root = image_tokens_root
        self.caption_return_policy = caption_return_policy

    def __getitem__(self, idx):
        sample_dict = super().__getitem__(idx)
        
        all_tokens_dict = get_image_tokens(self.image_tokens_root, idx, 0)
        captions = sample_dict['caption_options']
        captions = get_return_captions(captions, self.caption_return_policy)
        all_tokens_dict['captions'] = captions
        return all_tokens_dict