import random
import torch
from torch.nn import ConstantPad2d

def get_return_captions(captions, caption_return_policy):
    ret_captions = captions
    if caption_return_policy == 'random':
        ret_captions = random.choice(captions)
    elif caption_return_policy == 'first':
        ret_captions = captions[0]

    return ret_captions

def get_image_tokens(image_tokens_root, image_id, image_index):
    node_tokens = torch.load(f'{image_tokens_root}/{image_id}_{image_index}_node_tokens.pt', map_location = 'cpu')
    edge_tokens = torch.load(f'{image_tokens_root}/{image_id}_{image_index}_edge_tokens.pt', map_location = 'cpu')
    image_features = torch.load(f'{image_tokens_root}/{image_id}_{image_index}_image_features.pt', map_location = 'cpu')
    num_non_pad_tokens = node_tokens.shape[0] + edge_tokens.shape[0]
    num_nodes = node_tokens.shape[0]
    image_tokens = torch.cat([node_tokens, edge_tokens], dim=0)
    
    image_attention_mask = torch.load(f'{image_tokens_root}/{image_id}_{image_index}_attention_mask.pt', map_location = 'cpu')
    pad = 77 - image_tokens.shape[0]
    image_tokens = ConstantPad2d((0, 0, 0, pad), 0)(image_tokens)
    image_attention_mask = ConstantPad2d((0, pad, 0, pad), 0)(image_attention_mask)

    ret = {
        'image_tokens': image_tokens,
        'image_features': image_features,
        'num_non_pad_tokens': num_non_pad_tokens,
        'num_nodes': num_nodes,
        'image_attention_mask': image_attention_mask
    }
    return ret


def get_dataset(dataset_name,
                image_tokens_root = None,
                transform = None,
                with_image_tokens = False, 
                caption_return_policy = 'all'):
    if dataset_name == 'coco' or dataset_name == 'coco_val':
        from .coco import CocoImagesAndCaptions, CocoImageTokensAndCaptions
        annFile = '/fs/cml-datasets/coco/annotations/captions_train2017.json'
        image_root = '/fs/cml-datasets/coco/images/train2017/'
        if dataset_name == 'coco_val':
            annFile = '/fs/cml-datasets/coco/annotations/captions_val2017.json'
            image_root = '/fs/cml-datasets/coco/images/val2017/'
            
        if with_image_tokens:
            dataset = CocoImageTokensAndCaptions(root = image_root,
                                                 image_tokens_root = image_tokens_root,
                                                 annFile = annFile,
                                                 caption_return_policy=caption_return_policy)
        else:
            dataset = CocoImagesAndCaptions(root = image_root,
                                            annFile = annFile,
                                            transform = transform,
                                            caption_return_policy=caption_return_policy)

    elif dataset_name == 'aro_vgr':
        from .aro import VG_RelationImagesAndCaptions, VG_RelationImageTokensAndCaptions
        if with_image_tokens:
            dataset = VG_RelationImageTokensAndCaptions(download = False,
                                                        root_dir = '/cmlscratch/nehamk/datasets/aro',
                                                        image_tokens_root = image_tokens_root)
        else:
            dataset = VG_RelationImagesAndCaptions(image_preprocess = transform,
                                                    download = False,
                                                    root_dir = '/cmlscratch/nehamk/datasets/aro')

    elif dataset_name == 'aro_vga':
        from .aro import VG_AttributionImagesAndCaptions, VG_AttributionImageTokensAndCaptions
        if with_image_tokens:
            dataset = VG_AttributionImageTokensAndCaptions(download = False,
                                                           root_dir = '/cmlscratch/nehamk/datasets/aro',
                                                           image_tokens_root = image_tokens_root)
        else:
            dataset = VG_AttributionImagesAndCaptions(image_preprocess = transform,
                                                      download = False,
                                                      root_dir = '/cmlscratch/nehamk/datasets/aro')

    elif dataset_name == 'aro_coco_order':
        from .aro import COCO_OrderImagesAndCaptions, COCO_OrderImageTokensAndCaptions
        if with_image_tokens:
            dataset = COCO_OrderImageTokensAndCaptions(download = False,
                                                       root_dir = '/cmlscratch/nehamk/datasets/aro/coco',
                                                       image_tokens_root = image_tokens_root)
        else:
            dataset = COCO_OrderImagesAndCaptions(image_preprocess = transform,
                                                  download = False,
                                                  root_dir = '/cmlscratch/nehamk/datasets/aro/coco')

    elif dataset_name == 'winoground':
        from .winoground import WinogroundImagesAndCaptions, WinogroundImageTokensAndCaptions
        if with_image_tokens:
            dataset = WinogroundImageTokensAndCaptions(root = '/cmlscratch/nehamk/datasets/winoground',
                                                       image_tokens_root = image_tokens_root)
        else:
            dataset = WinogroundImagesAndCaptions(root = '/cmlscratch/nehamk/datasets/winoground',
                                                  transform = transform)

    else:
        raise ValueError(f'Dataset {dataset_name} not recognized')

    
    return dataset
        