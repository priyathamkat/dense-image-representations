import argparse
import pdb
import os
import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms

from torch_geometric.utils import to_networkx
import networkx as nx

from vision.vision_graph_constructor import VisionGraphConstructor
from vision.image_segmentor import ImageSegmentor

from text.text_graph_constructor import TextGraphConstructor

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


def visualize_graph(graph_obj, save_name):
    node_names = dict([(i, graph_obj.node_names[i]) for i in range(len(graph_obj.node_names))])
    edge_names = dict([(tuple(graph_obj.edge_index.T[i].numpy()), graph_obj.edge_names[i]) for i in range(len(graph_obj.edge_names))])
    print(edge_names)

    # Draw and save graph
    nx_graph = to_networkx(graph_obj)
    pos = nx.circular_layout(nx_graph, scale = 0.2)
    plt.figure(figsize = (5, 3))
    nx.draw(nx_graph, pos, labels = node_names, with_labels = True, font_size = 15, node_color = 'white')
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels = edge_names, font_size = 15)
    plt.savefig(save_name, format='PNG')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--visual_graphs_save_path', type=str)
    parser.add_argument('--parsed_captions_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    
    if not os.path.exists(f'{args.visual_graphs_save_path}'):
        os.makedirs(f'{args.visual_graphs_save_path}')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.PILToTensor()])

    train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/train2017/', 
                               annFile = '/fs/cml-datasets/coco/annotations/captions_train2017.json', 
                               transform = transform)
    val_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/val2017/',
                             annFile = '/fs/cml-datasets/coco/annotations/captions_val2017.json', 
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
    vision_graph_constructor = VisionGraphConstructor(pretrained_ram_model_path='pretrained_checkpoints')

    text_graph_constructor = TextGraphConstructor()

    device = torch.device('cuda')

    for i, batch in enumerate(train_loader):
        # One image at a time, Batch size = 1
        images, image_sizes, image_ids = batch 

        parsed_captions = glob.glob(f'{args.parsed_captions_path}/{image_ids[0].item()}_*.json')
        parsed_captions = [json.load(open(p)) for p in parsed_captions]
        text_graphs = [text_graph_constructor(c) for c in parsed_captions]

        inst_seg = segmentor(images[0].to(device), image_sizes[0])

        pil_resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
        ])

        original_image = pil_resize_transform(images[0])
        visual_graph = vision_graph_constructor(original_image, inst_seg)

        # [visualize_graph(text_graphs[t], f'graph_vis/{image_ids[0].item()}_{t}_text_graph.png') for t in range(len(text_graphs))]
        # visualize_graph(visual_graph, f'graph_vis/{image_ids[0].item()}_visual_graph.png')
        # pdb.set_trace()
        # torch.save(visual_graph, f'{args.visual_graphs_save_path}/{image_ids[0].item()}_graph.pt')