# from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor

from pathlib import Path
import matplotlib.pyplot as plt
import pprint
from tqdm import tqdm

import numpy as np
from PIL import Image

from detectron2.data.detection_utils import read_image
from detectron2.utils.colormap import colormap
import json

from vision.seem_module.modeling.BaseModel import BaseModel
from vision.seem_module.modeling import build_model
from vision.seem_module.utils.arguments import load_opt_from_config_files
from vision.seem_module.utils.constants import COCO_PANOPTIC_CLASSES
from vision.seem_module.utils.distributed import init_distributed
from vision.seem_module.utils.visualizer import Visualizer

import os
import torch 
from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from PIL import Image
from torchvision import transforms 
import numpy as np 
import matplotlib.pyplot as plt 
from mmengine.config import Config
import pickle as pkl

def rgb2id(color):
	if isinstance(color, np.ndarray) and len(color.shape) == 3:
		if color.dtype == np.uint8:
			color = color.astype(np.int32)
		return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
	return int(color[0] + 256 * color[1] + 256 * 256 * color[2])



conf_files = '/cmlscratch/snawathe/dense-image-representations/dense-image-representations/vision/seem_module/configs/seem/focall_unicl_lang_demo.yaml'
opt = load_opt_from_config_files([conf_files])
opt = init_distributed(opt)

# META DATA
cur_model = 'None'
if 'focalt' in conf_files:
	pretrained_pth = os.path.join("seem_focalt_v0.pt")
	if not os.path.exists(pretrained_pth):
		os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt"))
	cur_model = 'Focal-T'
elif 'focal' in conf_files:
	pretrained_pth = os.path.join("seem_focall_v1.pt")
	if not os.path.exists(pretrained_pth):
		os.system("wget {}".format("https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v1.pt"))
	cur_model = 'Focal-L'

'''
build model
'''
model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
with torch.no_grad():
	thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(COCO_PANOPTIC_CLASSES))]
	thing_dataset_id_to_contiguous_id = {x:x for x in range(len(COCO_PANOPTIC_CLASSES))}
	
	MetadataCatalog.get("demo").set(
		thing_colors=thing_colors,
		thing_classes=COCO_PANOPTIC_CLASSES,
		thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
	)
	model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ["background"], is_eval=False)
	metadata = MetadataCatalog.get('demo')
	model.model.metadata = metadata
	model.model.sem_seg_head.num_classes = len(COCO_PANOPTIC_CLASSES)



transform = transforms.Compose([
				transforms.Resize((512, 512), interpolation=Image.BICUBIC),
				transforms.PILToTensor()
			])

def area(t) -> float:
	return (t != 0).sum()

def iou(mask1, mask2):
	intersection = np.logical_and(mask1, mask2)
	union = np.logical_or(mask1, mask2)
	iou_score = np.sum(intersection) / np.sum(union)
	return iou_score



with open('/cmlscratch/snawathe/dense-image-representations/dense-image-representations/vision/relate_anything/psg.json') as f:
    psg_dataset_file = json.load(f)

psg_thing_cats = psg_dataset_file['thing_classes']
psg_stuff_cats = psg_dataset_file['stuff_classes']
psg_obj_cats = psg_thing_cats + psg_stuff_cats
psg_rel_cats = psg_dataset_file['predicate_classes']
psg_dataset = {d["image_id"]: d for d in psg_dataset_file['data']}
# psg_dataset_coco_id = {d["coco_image_id"]: d for d in psg_dataset_file['data']}

print('Number of images: {}'.format(len(psg_dataset)))
print('# Object Classes: {}'.format(len(psg_obj_cats)))
print('# Relation Classes: {}'.format(len(psg_rel_cats)))



def process_image(img_id: int):
	data = psg_dataset[img_id]
     
	image = read_image(f"/fs/cml-datasets/coco/images/{data['file_name']}", format="RGB")
	seg_map = read_image(f"/cmlscratch/snawathe/dense-image-representations/coco/{data['pan_seg_file_name']}", format="RGB")
	seg_map = rgb2id(seg_map)
    
	# get seperate masks
	gt_masks = []
	labels_coco = []
	for i, s in enumerate(data["segments_info"]):
		label = psg_obj_cats[s["category_id"]]
		labels_coco.append(label)
		gt_masks.append(seg_map == s["id"])
    
	# run SEEM
	image_ori = Image.open(f"/fs/cml-datasets/coco/images/{data['file_name']}").convert("RGB")
	batch_inputs = [{
		'image': transform(image_ori),
		'height': image_ori.size[1],
		'width': image_ori.size[0]
	}]
	with torch.no_grad():
		outputs = model.forward(batch_inputs)
	instances = []
	for i in range(100):
		instances.append({
			'pred_mask_embs': outputs[0]['instances'].get_fields()['pred_mask_embs'][i].cpu().numpy(),
			'pred_masks': outputs[0]['instances'].get_fields()['pred_masks'][i].cpu().numpy(),
			'pred_boxes': outputs[0]['instances'].get_fields()['pred_boxes'][i],
			'scores': outputs[0]['instances'].get_fields()['scores'][i].cpu().numpy(),
			'pred_classes': outputs[0]['instances'].get_fields()['pred_classes'][i].cpu().numpy()
		})
	
	# sort and filter instances
	sorted_instances = sorted(instances, key=lambda x: area(x['pred_masks']), reverse=True)
	filtered_instances = []
	for instance in sorted_instances:
		if instance['scores'] < 0.5:
			continue
		duplicate = False
		for filtered_mask in filtered_instances:
			if iou(instance['pred_masks'], filtered_mask['pred_masks']) > 0.5:
				duplicate = True
				break

		if not duplicate:
			filtered_instances.append(instance)
	ddup_instances = filtered_instances

	# GT matching
	gt_index_mgt = []
	for mask_id, mask_dict in enumerate(ddup_instances):
		max_iou = 0
		for gt_mask_id, gt_mask in enumerate(gt_masks):
			current_iou = iou(gt_mask, mask_dict['pred_masks'])
			if current_iou > max_iou and current_iou > 0.6:
				max_iou = current_iou
				gt_index_mgt.append({gt_mask_id: mask_id})

	gt_index_gtm = []
	for gt_mask_id, gt_mask in enumerate(gt_masks):
		max_iou = 0
		for mask_id, mask_dict in enumerate(ddup_instances):
			current_iou = iou(gt_mask, mask_dict['pred_masks'])
			if current_iou > max_iou and current_iou > 0.5:
				max_iou = current_iou
				gt_index_gtm.append({gt_mask_id: mask_id})
				
	common = [x for x in gt_index_gtm if x in gt_index_mgt]

	# convert gt_index to a dictionary
	gt_dict = {}
	for d in common:
		gt_dict.update(d)
	gt_list = list(gt_dict.keys())

	# relation idx swapping
	relations = data['relations'].copy()
	new_relations = []
	for sublist in relations:
		if all(x in gt_dict for x in sublist[:-1]):
			new_sublist = [gt_dict[x] for x in sublist[:-1]] + [sublist[-1]] 
			new_relations.append(new_sublist)
	
	print(f"{img_id=}, {len(filtered_instances)=}, {len(gt_masks)=}, {len(common)=}, {len(relations)=}, {len(new_relations)=}")

	ddup_feat = np.array([instance['pred_mask_embs'] for instance in ddup_instances])

	save_entry = {
		'id': img_id,
		'feat': ddup_feat,
		'relations': new_relations,
		'is_train': img_id not in psg_dataset_file['test_image_ids'],
	}
	np.savez(f'/cmlscratch/snawathe/dense-image-representations/ram_dataset/{img_id}.npz', **save_entry)



for i, img_id in enumerate(psg_dataset.keys()):
	process_image(img_id)
	if i % 100 == 0:
		print(f"{i=}", flush=True)
