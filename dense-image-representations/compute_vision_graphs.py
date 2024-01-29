import argparse
import pdb
import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from vision.vision_graph_constructor import VisionGraphConstructor
from vision.image_segmentor import ImageSegmentor


class CocoImages(CocoDetection):
    def __init__(self, root, annFile, transform):
        super().__init__(root, annFile, transform)
        
    def __getitem__(self, index):
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        im_size = np.array(image.size)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, im_size, id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--visual_graphs_save_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    
    if not os.path.exists(f'{args.visual_graphs_save_path}'):
        os.makedirs(f'{args.visual_graphs_save_path}')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.PILToTensor()])

    train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/train2017/', 
                               annFile = '/fs/cml-datasets/coco/annotations/instances_train2017.json', 
                               transform = transform)
    val_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/val2017/',
                             annFile = '/fs/cml-datasets/coco/annotations/instances_val2017.json', 
                             transform = transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle = False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    segmentor = ImageSegmentor(pretrained_model_path='pretrained_checkpoints', seem_config = 'vision/seem_module/configs/seem/focall_unicl_lang_demo.yaml')
    graph_constructor = VisionGraphConstructor(pretrained_ram_model_path='pretrained_checkpoints')
    device = torch.device('cuda')

    for i, batch in enumerate(train_loader):
        # One image at a time, Batch size = 1
        images, image_sizes, image_ids = batch

        inst_seg = segmentor(images[0].to(device), image_sizes[0])

        pil_resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
        ])

        original_image = pil_resize_transform(images[0])
        graph_data = graph_constructor(original_image, inst_seg)

        torch.save(graph_data, f'{args.visual_graphs_save_path}/{image_ids[0].item()}_graph.pt')