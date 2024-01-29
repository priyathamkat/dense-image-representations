import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from detectron2.utils.colormap import random_color
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks

from seem_module.modeling.BaseModel import BaseModel
from seem_module.modeling import build_model
from seem_module.utils.arguments import load_opt_from_config_files
from seem_module.utils.constants import COCO_PANOPTIC_CLASSES
from seem_module.utils.distributed import init_distributed
from seem_module.utils.visualizer import Visualizer


class ImageSegmentor:
    def __init__(self, pretrained_model_path='', seem_config='seem_module/configs/seem/focall_unicl_lang_demo.yaml', device='cuda'):
        self.seem_config = seem_config
        seem_model, self.seem_metadata = self.load_seem_model(pretrained_model_path)
        self.device = device
        self.seem_model = seem_model.to(device)
        self.seem_model.eval()
        self.resize_pil_to_tensor = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.BICUBIC),
            transforms.PILToTensor()])
    
    def load_seem_model(self, pretrained_model_path):
        """"Returns a SEEM Model object and metadata for assigning labels for masks."""

        conf_files = self.seem_config
        opt = load_opt_from_config_files([conf_files])
        opt = init_distributed(opt)

        # META DATA
        cur_model = 'None'
        if 'focalt' in conf_files:
            pretrained_pth = os.path.join(pretrained_model_path, 'seem_focalt_v0.pt')
            if not os.path.exists(pretrained_pth):
                os.system('wget {}'.format('https://huggingface.co/xdecoder/SEEM/resolve/main/seem_focalt_v0.pt'))
            cur_model = 'Focal-T'
        elif 'focal' in conf_files:
            pretrained_pth = os.path.join(pretrained_model_path, 'seem_focall_v1.pt')
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

    def segment(self, pil_image):
        """Accepts a pil image and returns the detection Instances using SEEM. """
        
        image = self.resize_pil_to_tensor(pil_image).to(self.device)
        return self.__call__(image, pil_image.size)

    def __call__(self, image, image_size):
        """Accepts a tensor (512x512) image and returns the detection Instances using SEEM.

        Args:
            image: torch.Tensor of shape (3, 512, 512)
            image_size: List or tuple (width, height)
        """
        
        batched_inputs = [{'image': image, 'height': image_size[1], 'width': image_size[0]}]
        
        with torch.no_grad():
            seem_outputs = self.seem_model.forward(batched_inputs)
        
        inst_seg = seem_outputs[0]['instances']
        
        sel_inst_seg = inst_seg[(inst_seg.scores > 0.5).cpu()]
        
        sel_inst_seg.pred_masks = sel_inst_seg.pred_masks.cpu()
        sel_inst_seg.pred_boxes = BitMasks(sel_inst_seg.pred_masks > 0).get_bounding_boxes()

        return sel_inst_seg

    def visualize_segmented_image(self, pil_image, inst_seg):
        """Visualizes the instance segmentation on the current image and saves it as inst.png."""

        visual = Visualizer(np.asarray(pil_image), metadata=self.seem_metadata)
        demo = visual.draw_instance_predictions(inst_seg) # rgb Image

        demo.save('inst.png')

        # pred_to_box = {}
        # pred_to_mask = {}
        # for b in range(bboxes.tensor.shape[0]):
        #     pred_cl = sel_inst_seg.pred_classes[b].item()
        #     box = bboxes[b].tensor.long().squeeze()
        #     mask = masks[b].tensor.squeeze()
        #     if pred_cl not in pred_to_box:
        #         pred_to_box[pred_cl] = torch.zeros(original_image.shape).cuda()
        #         pred_to_mask[pred_cl] = torch.zeros(original_image.shape).cuda()

        #     pred_to_mask[pred_cl] = torch.maximum(pred_to_mask[pred_cl], (original_image * mask.long().cuda())/255)
        #     pred_to_box[pred_cl][:, box[1]: box[3], box[0]: box[2]] = original_image[:, box[1]: box[3], box[0]: box[2]]/255
        
        
        # print([COCO_PANOPTIC_CLASSES[k] for k in pred_to_box])
        # save_image(torch.stack(list(pred_to_box.values())), 'boxes.png')
        # save_image(torch.stack(list(pred_to_mask.values())), 'masks.png')
