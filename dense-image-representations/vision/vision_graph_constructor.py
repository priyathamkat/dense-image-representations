import os 
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt 

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx

from seem_module.utils.constants import COCO_PANOPTIC_CLASSES

from relate_anything.segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
from relate_anything.utils import relation_classes
from relate_anything.ram_train_eval import RamModel, RamPredictor
from mmengine.config import Config

from image_segmentor import ImageSegmentor

from transformers import AutoTokenizer, T5EncoderModel

import pdb

class Predictor(RamPredictor):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self._build_model()

    def _build_model(self):
        self.model = RamModel(**self.config.model).to(self.device)
        if self.config.load_from is not None:
            self.model.load_state_dict(torch.load(self.config.load_from), strict=False)


class VisionGraphConstructor:
    def __init__(self, pretrained_ram_model_path='', device = 'cuda'):
        self.sam_predictor, self.ram_predictor = self.load_ram_model(pretrained_ram_model_path, device)
        self.sam_predictor.model = self.sam_predictor.model.to(device)
        self.ram_predictor.model = self.ram_predictor.model.to(device)
        self.sam_predictor.model.eval()
        self.ram_predictor.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.t5_model = T5EncoderModel.from_pretrained("t5-small").to(device)

    def load_ram_model(self, pretrained_ram_model_path, device):
        sam = build_sam(checkpoint = os.path.join(pretrained_ram_model_path, 'sam_vit_h_4b8939.pth'))
        sam_predictor = SamPredictor(sam)

        # load ram model
        model_path = os.path.join(pretrained_ram_model_path, 'epoch_12.pth')
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

        ram_predictor = Predictor(config, device)
        return sam_predictor, ram_predictor

    def get_ram_relationship(self, bbox1, bbox2):
        """Provided two bounding boxes, return the relationship index and score."""
        with torch.no_grad():
            mask1, score1, logit1, feat1 = self.sam_predictor.predict(
                # mask_input = masks[2].tensor,
                box = bbox1.tensor.cpu().numpy(),
                multimask_output = False
            )

            mask2, score2, logit2, feat2 = self.sam_predictor.predict(
                # mask_input = masks[2].tensor,
                box = bbox2.tensor.cpu().numpy(),
                multimask_output = False
            )

            feat = torch.cat((feat1, feat2), dim=1)
            matrix_output, rel_triplets = self.ram_predictor.predict(feat)
            
        subject_output = matrix_output.permute([0,2,3,1])[:,0,1:]
        
        output = subject_output[:,0]
        topk_indices = torch.argsort(-output).flatten()
        logit_score = -torch.sort(-output).values.flatten()[:1].item()
        # relation = relation_classes[topk_indices[:1][0]]
        return topk_indices[:1][0], logit_score


    def __call__(self, pil_image, inst_seg):
        """Returns a graph data object using detectron instances and RAM relationships."""
        
        bboxes = inst_seg.pred_boxes
        original_image = np.asarray(pil_image)
        
        # Extract relationships in the format [[source_node_index, target_node_index], [source_node_index, target_node_index], ...]
        self.sam_predictor.set_image(original_image)
        edge_index = []
        edge_attr = []
        for b1 in range(bboxes.tensor.shape[0]):
            for b2 in range(bboxes.tensor.shape[0]):
                if b1 == b2:
                    continue
                relation, score = self.get_ram_relationship(bboxes[b1], bboxes[b2])
                if score > 0.005:
                    edge_index.append([b1, b2])
                    edge_attr.append(relation)
        
        node_classes = [COCO_PANOPTIC_CLASSES[c.item()] for c in inst_seg.pred_classes.cpu()]
        edge_classes = [relation_classes[c.item()] for c in edge_attr]

        graph_data = Data(x = inst_seg.pred_mask_embs.cpu(),
                        node_attr = self.encode_with_lm(node_classes),
                        node_names = node_classes,
                        edge_index = torch.tensor(edge_index).t(),
                        edge_attr = self.encode_with_lm(edge_classes),
                        edge_names = edge_classes)

        return graph_data

    def encode_with_lm(self, texts):
        """Accepts a list of texts and returns a list of encoded texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        inputs = inputs.cuda()
        outputs = self.t5_model(inputs).last_hidden_state
        return outputs.mean(dim=1)

    def visualize_graph(self, graph_obj):
        node_names = dict([(i, graph_obj.node_names[i]) for i in range(len(graph_obj.node_attr))])
        edge_names = dict([(tuple(graph_obj.edge_index.T[i].numpy()), graph_obj.edge_names[i]) for i in range(len(graph_obj.edge_attr))])
        print(edge_names)

        # Draw and save graph
        nx_graph = to_networkx(graph_obj)
        pos = nx.circular_layout(nx_graph)
        nx.draw(nx_graph, pos, labels = node_names, with_labels = True)
        nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels = edge_names, font_size = 8)
        plt.savefig('graph.png', format='PNG')


if __name__ == '__main__':
    segmentor = ImageSegmentor(pretrained_model_path='../pretrained_checkpoints')
    pil_image = Image.open('../person_with_coffee.jpeg').convert('RGB')
    inst_seg = segmentor.segment(pil_image)
    print([COCO_PANOPTIC_CLASSES[c] for c in inst_seg.pred_classes])
    segmentor.visualize_segmented_image(pil_image, inst_seg)

    graph_constructor = VisionGraphConstructor(pretrained_ram_model_path='../pretrained_checkpoints')
    graph_obj = graph_constructor(pil_image, inst_seg)
    pdb.set_trace()
    graph_constructor.visualize_graph(graph_obj)

    
