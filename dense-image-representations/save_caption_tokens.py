import json
import os
import pdb
import torch
import argparse
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='coco')

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
dataset = args.dataset

save_path = f'{dataset}_text_tokens'
if not os.path.exists(save_path):
    os.makedirs(save_path)

if 'coco' in dataset and 'aro' not in dataset:
    f = open('/fs/cml-datasets/coco/annotations/captions_train2017.json')
    if dataset == 'coco_val':
        f = open('/fs/cml-datasets/coco/annotations/captions_val2017.json')

    d = json.load(f)['annotations']

    for sam in d:
        image_id = sam['image_id']
        cap_id = sam['id']
        caption = sam['caption']
        tokens = tokenizer(caption, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.squeeze().cpu()
        
        torch.save(tokens, f'{save_path}/{image_id}_{cap_id}_tokens.pt')

elif dataset == 'winoground':
    from datasets import load_dataset

    winoground_dataset = load_dataset('facebook/winoground', use_auth_token='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', cache_dir = '/cmlscratch/nehamk/datasets/winoground')
    data = winoground_dataset['test']

    for i in range(len(data)):
        image_id = i
        caption_0 = data[i]['caption_0']
        caption_1 = data[i]['caption_1']

        tokens_0 = tokenizer(caption_0, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.squeeze().cpu()    
        tokens_1 = tokenizer(caption_1, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.squeeze().cpu()    

        torch.save(tokens_0, f'{save_path}/{image_id}_0_tokens.pt')
        torch.save(tokens_1, f'{save_path}/{image_id}_1_tokens.pt')


elif 'aro' in dataset:
    import sys
    sys.path.append('/cmlscratch/nehamk/vision-language-models-are-bows')
    from dataset_zoo import VG_Relation, VG_Attribution, COCO_Order
    
    root_dir="/cmlscratch/nehamk/datasets/aro"

    if dataset == 'aro_vgr':
        data = VG_Relation(image_preprocess=None, download=False, root_dir=root_dir)
        
    elif dataset == 'aro_vga':
        data = VG_Attribution(image_preprocess=None, download=False, root_dir=root_dir)

    elif dataset == 'aro_coco_order':
        data = COCO_Order(image_preprocess=None, download=False, root_dir=root_dir + '/coco', split='val')

    for i in range(len(data)):
        caption_options = data[i]['caption_options']
        tokens = tokenizer(caption_options, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.squeeze().cpu()

        for j in range(len(caption_options)):
            torch.save(tokens[j], f'{save_path}/{i}_{j}_tokens.pt')

        