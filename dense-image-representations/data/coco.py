import re
import glob
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets.coco import CocoDetection
import numpy as np

text_tokenizer_type_str = {
    't5_small': '',
    't5_base': '_t5_base',
    'clip': '_clip'
}

class CocoImages(CocoDetection):
    def __init__(self, root, annFile, transform):
        super().__init__(root, annFile, transform)
        self.annFile = annFile
        
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        im_size = np.array(image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, im_size, id


class CocoImagesAndTextTokensForViT(Dataset):
    def __init__(self, image_root, image_annFile, vit_processor, text_root, text_tokenizer_type='t5_small'):
        self.image_root = image_root
        self.text_root = text_root
        
        self.vit_processor = vit_processor

        self.coco_images = CocoImages(image_root, image_annFile, None)

        self.text_tokenizer_str = text_tokenizer_type_str[text_tokenizer_type]

        saved_ids = [re.findall(r'(\d+)_(\d+)', i)[0] for i in glob.glob(f'{text_root}/*{self.text_tokenizer_str}_tokens.pt')]
        saved_ids_dict = {}
        for i in saved_ids:
            if int(i[0]) not in saved_ids_dict:
                saved_ids_dict[int(i[0])] = []
            saved_ids_dict[int(i[0])].append(int(i[1]))

        self.saved_ids = list(saved_ids_dict.keys())
        self.saved_ids_dict = saved_ids_dict
    
    def __len__(self):
        return len(self.coco_images)
    
    def __getitem__(self, idx):
        image, _, id = self.coco_images[idx]

        vit_inputs = self.vit_processor(image)

        if not isinstance(vit_inputs, torch.Tensor):
            vit_inputs = vit_inputs['pixel_values'][0]

        text_tokens = torch.load(f'{self.text_root}/{id}_{random.choice(self.saved_ids_dict[id])}{self.text_tokenizer_str}_tokens.pt', map_location='cpu')

        return vit_inputs, text_tokens
