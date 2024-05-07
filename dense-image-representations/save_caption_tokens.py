import json
import os
import pdb
import torch
import argparse
import clip
from transformers import AutoTokenizer

def tokenize(captions, tokenizer_type, tokenizer=None):
    tokens = None
    if tokenizer_type == 'clip':
        tokens = clip.tokenize(captions).squeeze().cpu()
    else:
        tokens = tokenizer(captions, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.squeeze().cpu()
    return tokens


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='coco')
parser.add_argument('--tokenizer_type', type=str, default='clip')

args = parser.parse_args()

tokenizer = None
if args.tokenizer_type == 't5_small':
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
elif args.tokenizer_type == 't5_base':
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

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
        tokens = tokenize(caption, args.tokenizer_type, tokenizer)
        
        torch.save(tokens, f'{save_path}/{image_id}_{cap_id}_{args.tokenizer_type}_tokens.pt')
        os.chmod(f'{save_path}/{image_id}_{cap_id}_{args.tokenizer_type}_tokens.pt', 0o0777)

elif dataset == 'winoground':
    from datasets import load_dataset

    winoground_dataset = load_dataset('facebook/winoground', use_auth_token='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', cache_dir = '/cmlscratch/nehamk/datasets/winoground')
    data = winoground_dataset['test']

    for i in range(len(data)):
        image_id = i
        caption_0 = data[i]['caption_0']
        caption_1 = data[i]['caption_1']

        tokens_0 = tokenize(caption_0, args.tokenizer_type, tokenizer)   
        tokens_1 = tokenize(caption_1, args.tokenizer_type, tokenizer)  

        torch.save(tokens_0, f'{save_path}/{image_id}_0_{args.tokenizer_type}_tokens.pt')
        torch.save(tokens_1, f'{save_path}/{image_id}_1_{args.tokenizer_type}_tokens.pt')

        os.chmod(f'{save_path}/{image_id}_0_{args.tokenizer_type}_tokens.pt', 0o0777)
        os.chmod(f'{save_path}/{image_id}_1_{args.tokenizer_type}_tokens.pt', 0o0777)


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
        tokens = tokenize(caption_options, args.tokenizer_type, tokenizer)

        for j in range(len(caption_options)):
            torch.save(tokens[j], f'{save_path}/{i}_{j}_{args.tokenizer_type}_tokens.pt')
            
            os.chmod(f'{save_path}/{i}_{j}_{args.tokenizer_type}_tokens.pt', 0o0777)

        