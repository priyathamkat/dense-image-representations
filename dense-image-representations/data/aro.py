from torch.utils.data import Dataset
import sys
sys.path.append('/cmlscratch/nehamk/vision-language-models-are-bows')
from dataset_zoo import VG_Relation, VG_Attribution

import numpy as np

class ARORelation(Dataset):
    def __init__(self, root, transform=None):
        self.data = VG_Relation(image_preprocess=None, download=True, root_dir=root)
        self.transform = transform

    def __len__(self):
        # return len(self.data)
        return 10

    def __getitem__(self, idx):
        sample_dict = self.data[idx]

        image = sample_dict['image_options'][0]

        im_size = np.array(image.size)
        # cap0 = sample_dict['caption_0']
        # cap1 = sample_dict['caption_1']
        if self.transform:
            image = self.transform(image)

        return image, im_size, idx