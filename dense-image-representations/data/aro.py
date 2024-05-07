from torch.utils.data import Dataset
import sys
sys.path.append('/cmlscratch/nehamk/vision-language-models-are-bows')
from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order
import torch
import numpy as np
import glob

text_tokenizer_type_str = {
    't5_small': '',
    't5_base': '_t5_base',
    'clip': '_clip'
}


class ARO(Dataset):
    def __init__(self, root, transform=None, task='aro_vgr'):
        if task == 'aro_vgr':
            self.data = VG_Relation(image_preprocess=None, download=True, root_dir=root)
        elif task == 'aro_vga':
            self.data = VG_Attribution(image_preprocess=None, download=True, root_dir=root)
        elif task == 'aro_coco_order':
            self.data = COCO_Order(image_preprocess=None, download=True, root_dir=root + '/coco', split='val')
        else:
            raise ValueError('Invalid task')

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        image = sample_dict['image_options'][0]

        im_size = np.array(image.size)
        if self.transform:
            image = self.transform(image)

        return image, im_size, idx


class AROImagesAndCaptionTokens(Dataset):
    def __init__(self, root, text_tokens_root, vit_processor=None, task='aro_vgr', text_tokenizer_type='t5_small'):
        if task == 'aro_vgr':
            self.data = VG_Relation(image_preprocess=None, download=True, root_dir=root)
        elif task == 'aro_vga':
            self.data = VG_Attribution(image_preprocess=None, download=True, root_dir=root)
        elif task == 'aro_coco_order':
            self.data = COCO_Order(image_preprocess=None, download=True, root_dir=root + '/coco', split='val')
        else:
            raise ValueError('Invalid task')

        self.vit_processor = vit_processor

        self.text_tokens_root = text_tokens_root
        self.text_tokenizer_str = text_tokenizer_type_str[text_tokenizer_type]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        image = sample_dict['image_options'][0]

        text_tokens = [torch.load(f, map_location='cpu') for f in sorted(glob.glob(f'{self.text_tokens_root}/{idx}_*{self.text_tokenizer_str}_tokens.pt'))]

        if self.vit_processor:
            image = self.vit_processor(image)
            if not isinstance(image, torch.Tensor):
                image = image['pixel_values'][0]

        return image, text_tokens