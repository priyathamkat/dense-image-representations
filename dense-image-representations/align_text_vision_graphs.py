import argparse
import pdb
import os
import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

from data.winoground import Winoground
from data.coco import CocoImages

from vision.vision_graph_constructor import VisionGraphConstructor
from vision.image_segmentor import ImageSegmentor

from text.text_graph_constructor import TextGraphConstructor

import torch.nn.functional as F


def visualize_graph(graph_obj, save_name, title = ''):
    node_names = dict([(i, graph_obj.node_names[i]) for i in range(len(graph_obj.node_names))])

    # Draw and save graph
    plt.figure(figsize = (6, 4))
    plt.title(title)
    nx_graph = to_networkx(graph_obj)
    pos = nx.circular_layout(nx_graph)
    nx.draw(nx_graph, pos, labels = node_names, with_labels = True, font_size = 15, node_color = 'white')
    if graph_obj.edge_index.shape[0] > 0:
        edge_names = dict([(tuple(graph_obj.edge_index.T[i].numpy()), graph_obj.edge_names[i]) for i in range(len(graph_obj.edge_names))])
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels = edge_names, font_size = 15)
    plt.margins(x = 0.2)
    plt.savefig(save_name, format='PNG')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--visual_graphs_save_path', type=str)
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--parsed_captions_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    
    if not os.path.exists(f'{args.visual_graphs_save_path}'):
        os.makedirs(f'{args.visual_graphs_save_path}')
    
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.PILToTensor()])

    if args.dataset == 'coco':
        train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/train2017/', 
                                annFile = '/fs/cml-datasets/coco/annotations/captions_train2017.json', 
                                transform = transform)
    
    else:
        train_dataset = Winoground(root = '/cmlscratch/nehamk/datasets',
                                transform = transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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

        j = 0
        for j in range(len(image_ids)):
            image, image_size, image_id = images[j].squeeze(), image_sizes[j].squeeze(), image_ids[j].squeeze()

            parsed_caption = json.load(open(f'{args.parsed_captions_path}/{image_id.item()}_{j}.json'))
            text_graph = text_graph_constructor(parsed_caption).cpu()
            visualize_graph(text_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}_text_graph.png', parsed_caption['caption'])

            # labels = []
            # for nn in text_graph.node_names:
            #     blob = TextBlob(nn)
            #     labels += [tag[0] for tag in blob.tags if tag[1] == 'NN']

            # labels = list(set(labels))
            # segmentor.seem_metadata = segmentor.get_metadata(labels, include_coco_classes = False)

            inst_seg = segmentor(image.to(device), image_size)

            pil_resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
            ])

            original_image = pil_resize_transform(image)
            # segmentor.visualize_segmented_image(original_image, inst_seg)

            original_image.save(f'{args.dataset}_graph_vis/{image_id.item()}_{j}_original_image.png')
            visual_graph = vision_graph_constructor(original_image, inst_seg).cpu()

            visualize_graph(visual_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}_visual_graph.png')

            # torch.save(visual_graph, f'{args.visual_graphs_save_path}/{image_ids[0].item()}_graph.pt')

            # Match the nodes of visual and text graph using T5 embeddings 
            visual_nodes = visual_graph.node_attr
            visual_nodes_norm = F.normalize(visual_nodes, dim = 1)
            text_nodes_norm = F.normalize(text_graph.x, dim = 1)
            sim = torch.matmul(visual_nodes_norm, text_nodes_norm.T)

            new_text_graph = Data(x = torch.Tensor([]), edge_index = torch.Tensor([]), node_names = [])
            added_text_nodes = []
            
            for idx in range(visual_graph.x.shape[0]):
                matches = torch.where(sim[idx] > 0.5)[0]
                # 1) If no text nodes match, add the node to text graph
                if matches.shape[0] == 0:
                    node_name = visual_graph.node_names[idx]
                    new_text_graph.x = torch.cat([new_text_graph.x, text_graph_constructor.encode_with_lm(node_name).cpu()])
                    new_text_graph.node_names.append(node_name)
                    
            visualize_graph(new_text_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}_new_text_graph.png', parsed_caption['caption'])
            
            # # 2) If single match, add node from original text graph
            # elif matches.shape[0] == 1:
            #     # 2.1) Multiple text nodes match the same visual node
            #     if matches[0] in added_text_nodes:
                    
            #     # 2.2) Visual node matches 1 text node
            #     else:
            #         new_text_graph.x = torch.cat([new_text_graph.x, text_graph.x[matches[0]]])
            #         new_text_graph.node_names.append(text_graph.node_names[matches[0]])
            #         added_text_nodes.append(matches[0])
            # # 3) If multiple matches...
            # else: 
            #     # If there is an edge, then we need to check iw9
            #     pass



