import argparse
import pdb
import os
from PIL import Image
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms
from torchvision.utils import save_image
from torch_geometric.data import Data

from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks

from seem_module.modeling.BaseModel import BaseModel
from seem_module.modeling import build_model
from seem_module.utils.arguments import load_opt_from_config_files
from seem_module.utils.constants import COCO_PANOPTIC_CLASSES
from seem_module.utils.distributed import init_distributed
from seem_module.utils.visualizer import Visualizer

from relate_anything.segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
from relate_anything.utils import relation_classes
from relate_anything.ram_train_eval import RamModel, RamPredictor
from mmengine.config import Config

class Predictor(RamPredictor):
    def __init__(self,config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from, map_location=self.device), strict=False)
        self.model.eval()


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


def load_seem_model():
    conf_files = 'seem_module/configs/seem/focall_unicl_lang_demo.yaml'
    opt = load_opt_from_config_files([conf_files])
    opt = init_distributed(opt)

    # META DATA
    cur_model = 'None'
    if 'focalt' in conf_files:
        pretrained_pth = os.path.join('seem_focalt_v0.pt')
        if not os.path.exists(pretrained_pth):
            os.system('wget {}'.format('https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt'))
        cur_model = 'Focal-T'
    elif 'focal' in conf_files:
        pretrained_pth = os.path.join('seem_focall_v1.pt')
        if not os.path.exists(pretrained_pth):
            os.system('wget {}'.format('https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focall_v0.pt'))
        cur_model = 'Focal-L'

    '''
    build model
    '''
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    with torch.no_grad():
        thing_colors = [random_color(rgb=True, maximum=255).astype(np.int32).tolist() for _ in range(len(COCO_PANOPTIC_CLASSES))]
        thing_dataset_id_to_contiguous_id = {x:x for x in range(len(COCO_PANOPTIC_CLASSES))}
        
        MetadataCatalog.get('demo').set(
            thing_colors=thing_colors,
            thing_classes=COCO_PANOPTIC_CLASSES,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        )
        # model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ['background'], is_eval=False)
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(COCO_PANOPTIC_CLASSES + ['background'], is_eval=True)
        metadata = MetadataCatalog.get('demo')
        model.model.metadata = metadata
        model.model.sem_seg_head.num_classes = len(COCO_PANOPTIC_CLASSES)

    return model, metadata


def load_ram_model(device):
    sam = build_sam(checkpoint='sam_vit_h_4b8939.pth').to(device)
    sam_predictor = SamPredictor(sam)

    # load ram model
    model_path = 'epoch_12.pth'
    config = dict(
        model=dict(
            pretrained_model_name_or_path='bert-base-uncased',
            load_pretrained_weights=False,
            num_transformer_layer=2,
            input_feature_size=256,
            output_feature_size=768,
            cls_feature_size=512,
            num_relation_classes=56,
            pred_type='attention',
            loss_type='multi_label_ce',
        ),
        load_from=model_path,
    )
    config = Config(config)

    ram_predictor = Predictor(config)
    return sam_predictor, ram_predictor


def seem_visualize(image, inst_seg, seem_metadata):
    pil_resize_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
    ])
    visual = Visualizer(np.asarray(pil_resize_transform(image).convert('RGB')), metadata=seem_metadata)
    inst_seg.pred_masks = inst_seg.pred_masks.cpu()
    inst_seg.pred_boxes = BitMasks(inst_seg.pred_masks > 0).get_bounding_boxes()
    demo = visual.draw_instance_predictions(inst_seg) # rgb Image

    demo.save('inst.png')


def get_ram_relationship(bbox1, bbox2, sam_predictor, ram_predictor):
    '''
    Provided two bounding boxes, return the relationship index and score
    '''
    with torch.no_grad():
        mask1, score1, logit1, feat1 = sam_predictor.predict(
            # mask_input = masks[2].tensor,
            box = bbox1.tensor.cpu().numpy(),
            multimask_output = False
        )

        mask2, score2, logit2, feat2 = sam_predictor.predict(
            # mask_input = masks[2].tensor,
            box = bbox2.tensor.cpu().numpy(),
            multimask_output = False
        )

        feat = torch.cat((feat1, feat2), dim=1)
        matrix_output, rel_triplets = ram_predictor.predict(feat)
        
    subject_output = matrix_output.permute([0,2,3,1])[:,0,1:]
    
    output = subject_output[:,0]
    topk_indices = torch.argsort(-output).flatten()
    logit_score = -torch.sort(-output).values.flatten()[:1].item()
    # relation = relation_classes[topk_indices[:1][0]]
    return topk_indices[:1][0], logit_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--visual_nodes_save_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()
    
    if not os.path.exists(f'{args.visual_nodes_save_path}'):
        os.makedirs(f'{args.visual_nodes_save_path}')

    train_transform =  transforms.Compose([
                transforms.Resize((512, 512), interpolation=Image.BICUBIC),
                transforms.PILToTensor()
            ])

    test_transform = transforms.Compose([
                transforms.Resize((512, 512), interpolation=Image.BICUBIC),
                transforms.PILToTensor(),
            ])
    
    train_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/train2017/', 
                               annFile = '/fs/cml-datasets/coco/annotations/instances_train2017.json',
                               transform = train_transform)
    val_dataset = CocoImages(root = '/fs/cml-datasets/coco/images/val2017/',
                             annFile = '/fs/cml-datasets/coco/annotations/instances_val2017.json',
                             transform = test_transform)

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

    device = torch.device('cuda')
    seem_model, seem_metadata = load_seem_model()
    seem_model = seem_model.to(device)
    seem_model.eval()

    sam_predictor, ram_predictor = load_ram_model(device)

    for i, batch in enumerate(train_loader):
        # One image at a time, Batch size = 1
        images, image_sizes, image_ids = batch

        # print(image_ids[0].item())
        images = images.to(device)
        batched_inputs = [{'image': images[j], 'height': image_sizes[j][1], 'width': image_sizes[j][0]} for j in range(images.shape[0])]

        with torch.no_grad():
            seem_outputs = seem_model.forward(batched_inputs)

        inst_seg=seem_outputs[0]['instances']

        # visualize instance-segmented image
        '''
        seem_visualize(image=images[0],
                       inst_seg=inst_seg,
                       seem_metadata=seem_metadata)
        '''

        sel_inst_seg = inst_seg[(inst_seg.scores > 0.5).cpu()]
        
        masks = BitMasks(sel_inst_seg.pred_masks > 0)
        bboxes = masks.get_bounding_boxes()
        pil_resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC),
        ])
        original_image = np.asarray(pil_resize_transform(images[0]))
        # Draw boxes and masks by class
        '''
        original_image = transforms.Resize(inst_seg.image_size, interpolation=Image.BICUBIC)(images[0])
        pred_to_box = {}
        pred_to_mask = {}
        for b in range(bboxes.tensor.shape[0]):
            pred_cl = sel_inst_seg.pred_classes[b].item()
            box = bboxes[b].tensor.long().squeeze()
            mask = masks[b].tensor.squeeze()
            if pred_cl not in pred_to_box:
                pred_to_box[pred_cl] = torch.zeros(original_image.shape).cuda()
                pred_to_mask[pred_cl] = torch.zeros(original_image.shape).cuda()

            pred_to_mask[pred_cl] = torch.maximum(pred_to_mask[pred_cl], (original_image * mask.long().cuda())/255)
            pred_to_box[pred_cl][:, box[1]: box[3], box[0]: box[2]] = original_image[:, box[1]: box[3], box[0]: box[2]]/255
        
        
        print([COCO_PANOPTIC_CLASSES[k] for k in pred_to_box])
        save_image(torch.stack(list(pred_to_box.values())), 'boxes.png')
        save_image(torch.stack(list(pred_to_mask.values())), 'masks.png')
        '''
        
        # np.save(f'{args.visual_nodes_save_path}/{image_ids[0].item()}_pred_mask_embs.npy', sel_inst_seg.pred_mask_embs.cpu().numpy())
        # np.save(f'{args.visual_nodes_save_path}/{image_ids[0].item()}_pred_masks.npy', sel_inst_seg.pred_masks.cpu().numpy())
        # np.save(f'{args.visual_nodes_save_path}/{image_ids[0].item()}_pred_classes.npy', sel_inst_seg.pred_classes.cpu().numpy())

        # Extract relationships in the format [[source_node_index, target_node_index], [source_node_index, target_node_index], ...]
        sam_predictor.set_image(original_image)
        edge_index = []
        edge_attr = []
        for b1 in range(bboxes.tensor.shape[0]):
            for b2 in range(bboxes.tensor.shape[0]):
                if b1 == b2:
                    continue
                relation, score = get_ram_relationship(bboxes[b1], bboxes[b2], sam_predictor, ram_predictor)
                if score > 0.01:
                    edge_index.append([b1, b2])
                    edge_attr.append(relation)

        graph_data = Data(x = sel_inst_seg.pred_mask_embs.cpu(),
                          node_attr = sel_inst_seg.pred_classes.cpu(),
                          edge_index = torch.tensor(edge_index).t(),
                          edge_attr = torch.tensor(edge_attr))

        torch.save(graph_data, f'{args.visual_nodes_save_path}/{image_ids[0].item()}_graph.pt')

        # print([COCO_PANOPTIC_CLASSES[i] for i in graph_data.node_attr])
        # print([relation_classes[i] for i in graph_data.edge_attr])
        # print(graph_data.edge_index)