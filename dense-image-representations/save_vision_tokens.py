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
from torchvision.ops import box_iou

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

from data.winoground import Winoground
from data.coco import CocoImages
from data.aro import ARO

from vision.vision_graph_constructor import VisionGraphConstructor
from vision.image_segmentor import ImageSegmentor
from vision.seem_module.utils.constants import COCO_PANOPTIC_CLASSES

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


def get_iou(mask1, mask2):
    """
    Inputs:
    mask1: NxHxW torch.float32. Consists of [0, 1]
    mask2: NxHxW torch.float32. Consists of [0, 1]
    Outputs:
    ret: NxM torch.float32. Consists of [0 - 1]
    """

    N, H, W = mask1.shape
    M, H, W = mask2.shape

    mask1 = mask1.view(N, H*W)
    mask2 = mask2.view(M, H*W)

    intersection = torch.matmul(mask1, mask2.t())

    area1 = mask1.sum(dim=1).view(1, -1)
    area2 = mask2.sum(dim=1).view(1, -1)

    union = (area1.t() + area2) - intersection

    ret = torch.where(
        union == 0,
        torch.tensor(0., device=mask1.device),
        intersection / union,
    )

    return ret


def calculate_bbox_distance(box1, box2):
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    return torch.norm(torch.tensor(center1) - torch.tensor(center2))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()
    
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.PILToTensor()])

    if args.dataset == 'coco':
        train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/train2017/', 
                                annFile = '/fs/cml-datasets/coco/annotations/captions_train2017.json', 
                                transform = transform)

    elif args.dataset == 'coco_val':
        train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/val2017/', 
                                annFile = '/fs/cml-datasets/coco/annotations/captions_val2017.json', 
                                transform = transform)
    
    elif args.dataset == 'winoground':
        train_dataset = Winoground(root = '/cmlscratch/nehamk/datasets/winoground',
                                transform = transform)

    elif 'aro' in args.dataset:
        train_dataset = ARO(root = '/cmlscratch/nehamk/datasets/aro', transform = transform, task = args.dataset)

    else:
        raise ValueError('Invalid dataset')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    segmentor = ImageSegmentor(pretrained_model_path='pretrained_checkpoints', seem_config = 'vision/seem_module/configs/seem/focall_unicl_lang_demo.yaml')
    vision_graph_constructor = VisionGraphConstructor(pretrained_ram_model_path='pretrained_checkpoints')

    text_graph_constructor = TextGraphConstructor()

    device = torch.device('cuda')

    # if not os.path.exists(f'{args.dataset}_graph_vis'):
    #     os.makedirs(f'{args.dataset}_graph_vis')
    
    # if not os.path.exists(f'{args.dataset}_visual_graph'):
    #     os.makedirs(f'{args.dataset}_visual_graph')

    if not os.path.exists(f'{args.dataset}_visual_tokens'):
        os.makedirs(f'{args.dataset}_visual_tokens')

    
    for i, batch in enumerate(train_loader):
        # One image at a time, Batch size = 1
        images, image_sizes, image_ids = batch 

        if 'winoground' in args.dataset:
            text_tokens = glob.glob(f'{args.dataset}_text_tokens/{image_ids[0].item()}_*.pt')
        else:
            text_tokens = [0]
        if len(text_tokens) == 0:
            continue
        j = 0
        for j in range(len(text_tokens)):
            # Multiple images per caption (winoground case)
            if len(images) > 1:
                image, image_size, image_id = images[j].squeeze(), image_sizes[j].squeeze(), image_ids[j].squeeze()
            else:
                image, image_size, image_id = images.squeeze(), image_sizes.squeeze(), image_ids.squeeze()
           
            if os.path.exists(f'{args.dataset}_visual_tokens/{image_id.item()}_{j}_attention_mask.pt'):
                continue
            
            # parsed_caption = json.load(open(parsed_captions[j]))
            # text_graph = text_graph_constructor(parsed_caption).cpu()
            # visualize_graph(text_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}_text_graph.png', parsed_caption['caption'])
            
            # segmentor.seem_metadata = segmentor.get_metadata(class_list = [])

            
            if len(image.shape) == 2:
                image = torch.stack([image, image, image], dim = 0)
            image = image[:3, :, :] # winoground case when there are 4 channels

            inst_seg = segmentor(image.to(device), image_size)

            pil_resize_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
            ])
            original_image = pil_resize_transform(image)
            # original_image.save(f'{args.dataset}_graph_vis/{image_id.item()}_{j}_original_image.png')
            # segmentor.visualize_segmented_image(original_image, inst_seg, save_path = f'{args.dataset}_graph_vis/{image_id.item()}_{j}_inst_seg.png')

            # We have (1) inst segmentations and (2) attributed objects from the parsed caption
            # Now we need to match (1) and (2) by asking an LLM 

            # # Get matches based on IOU
            # segmentor.seem_metadata = segmentor.get_metadata(labels, include_coco_classes = False)
            # text_graph_guided_inst_seg = segmentor(image.to(device), image_size)
            # segmentor.visualize_segmented_image(original_image, text_graph_guided_inst_seg, save_path = f'{args.dataset}_graph_vis/{image_id.item()}_{j}_text_graph_guided_inst_seg.png')

            # mask_iou_matrix = torch.zeros((text_graph_guided_inst_seg.pred_masks.shape[0], inst_seg.pred_masks.shape[0]))
            # for m1 in range(text_graph_guided_inst_seg.pred_masks.shape[0]):
            #     for m2 in range(inst_seg.pred_masks.shape[0]):
            #         mask_iou_matrix[m1][m2] = get_iou(text_graph_guided_inst_seg.pred_masks[m1].unsqueeze(0), inst_seg.pred_masks[m2].unsqueeze(0)).squeeze()

            # box_iou_matrix = box_iou(text_graph_guided_inst_seg.pred_boxes.tensor, inst_seg.pred_boxes.tensor)
            # matches = torch.where(mask_iou_matrix > 0.5)

            visual_graph = vision_graph_constructor(original_image, inst_seg).cpu()
            # visualize_graph(visual_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}.png')
            # torch.save(visual_graph, f'{args.dataset}_visual_graph/{image_id.item()}_{j}.pt')


            num_tokens = visual_graph.x.shape[0] + len(visual_graph.edge_names)
            attention_mask = torch.zeros((num_tokens, num_tokens))
            i = 0
            # attend only where necessary
            for edge in visual_graph.edge_index.T:
                attention_mask[edge[0], edge[1]] = 1 # directional
                attention_mask[edge[0], visual_graph.x.shape[0] + i] = 1 # node 1 to edge
                attention_mask[edge[1], visual_graph.x.shape[0] + i] = 1 # node 2 to edge
                attention_mask[visual_graph.x.shape[0] + i, edge[0]] = 1 # edge to node 1
                attention_mask[visual_graph.x.shape[0] + i, edge[1]] = 1 # edge to node 2

            bboxes = inst_seg.pred_boxes
            for b1 in range(bboxes.tensor.shape[0]):
                dists = []
                for b2 in range(bboxes.tensor.shape[0]):
                    dists.append(calculate_bbox_distance(bboxes.tensor[b1], bboxes.tensor[b2]))
                
                dists = torch.stack(dists)
                for idx in dists.sort().indices[:2]:
                    if idx == b1:
                        continue
                    attention_mask[b1, idx] = 1

            visual_tokens = torch.cat([visual_graph.x, visual_graph.edge_attr], dim = 0)
            torch.save(visual_tokens, f'{args.dataset}_visual_tokens/{image_id.item()}_{j}_tokens.pt')
            torch.save(attention_mask, f'{args.dataset}_visual_tokens/{image_id.item()}_{j}_attention_mask.pt')
            
            # Match the nodes of visual instances and text graph using T5 embeddings 
            # Optionally match based on the nouns in the text nodes
            # labels = []
            # for nn in text_graph.node_names:
            #     blob = TextBlob(nn)
            #     labels += [tag[0] for tag in blob.tags if tag[1] == 'NN']
            # labels = list(set(labels))
            # # Map labels to COCO_PANOPTIC_CLASSES
            # labels_emb = torch.Tensor(F.normalize(text_graph_constructor.encode_with_lm(labels)))
            # coco_classes_emb = torch.Tensor(F.normalize(text_graph_constructor.encode_with_lm(COCO_PANOPTIC_CLASSES)))
            # sim = torch.matmul(labels_emb, coco_classes_emb.T)


            # visual_nodes = torch.cat([text_graph_constructor.encode_with_lm(COCO_PANOPTIC_CLASSES[c.item()]).cpu() for c in inst_seg.pred_classes.cpu()])
            # visual_nodes_norm = F.normalize(visual_nodes, dim = 1)
            # text_nodes_norm = F.normalize(text_graph.x, dim = 1)
            # # measure similarity by cosine of T5 embeddings
            # sim = torch.matmul(visual_nodes_norm, text_nodes_norm.T)
            
            # new_text_graph = Data(x = torch.Tensor([[]]), edge_index = torch.Tensor([]), node_names = [])
            # added_text_nodes = []
            
            # for idx in range(inst_seg.pred_masks.shape[0]):
            #     matches = torch.where(sim[idx] > 0.5)[0]
            #     text_node_name = None
            #     # 1) If no text nodes match, add the node to text graph
            #     if matches.shape[0] == 0:
            #         text_node_name = COCO_PANOPTIC_CLASSES[inst_seg.pred_classes[idx].item()]
                                
            #     # 2) If single match, add node from original text graph
            #     elif matches.shape[0] == 1:
            #         # 2.1) Multiple text nodes match the same visual node
            #         if matches[0] in added_text_nodes:
            #             # rarely happens 
            #             print("Many text nodes matched the same visual node")
            #             pdb.set_trace()
            #         # 2.2) Visual node matches 1 text node
            #         else:
            #             text_node_name = text_graph.node_names[matches[0]]
            #             added_text_nodes.append(matches[0])
            #     # 3) If multiple matches...
            #     else: 
            #         # Compare IOUs of the visual node with the matched text nodes
            #         pass
                    

            # visualize_graph(new_text_graph, f'{args.dataset}_graph_vis/{image_id.item()}_{j}_new_text_graph.png', parsed_caption['caption'])
