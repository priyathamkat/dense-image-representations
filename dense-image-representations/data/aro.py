from torch.utils.data import Dataset
import sys
sys.path.append('/cmlscratch/nehamk/vision-language-models-are-bows')
from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order

import numpy as np

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
