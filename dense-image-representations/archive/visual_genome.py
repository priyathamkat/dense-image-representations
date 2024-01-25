import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5EncoderModel


class VisualGenomeRelationshipWithSAMAndT5(Dataset):
    def __init__(self, root_dir, sam_model, t5_tokenizer, t5_encoder, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # check if sam relationship emb df is already saved
        if os.path.exists(os.path.join(root_dir, 'relationships_dataset_with_emb.json')):
            self.data = pd.read_json(os.path.join(root_dir, 'relationships_dataset_with_emb.json'))
        else:
            self.sam_prompt_encoder = sam_model.prompt_encoder
            self.sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
            self.t5_tokenizer = t5_tokenizer
            self.t5_encoder = t5_encoder

            self.relationship_df = pd.read_json(os.path.join(root_dir, 'relationships.json'))
            self.image_data_df = pd.read_json(os.path.join(root_dir, 'image_data.json'))
            self.image_folder = os.path.join(root_dir, 'VG_100K')
            self.data = self._prepare_data()

    
    def _prepare_data(self):
        self.data = []
        for i, row in self.relationship_df.iterrows():
            print(i)
            for j, relationship in enumerate(row['relationships']):
                # Get bboxes in xyxy format
                object_bbox = np.array([relationship['object']['x'], 
                relationship['object']['y'], 
                relationship['object']['x'] + relationship['object']['w'], 
                relationship['object']['y'] + relationship['object']['h']])

                subject_bbox = np.array([relationship['subject']['x'], 
                relationship['subject']['y'], 
                relationship['subject']['x'] + relationship['subject']['w'], 
                relationship['subject']['y'] + relationship['subject']['h']])

                image_size = (self.image_data_df.loc[self.image_data_df['image_id'] == row['image_id']]['width'].item(), 
                self.image_data_df.loc[self.image_data_df['image_id'] == row['image_id']]['width'].item())
                
                # Prompt encoder output is of format: (sparse embeddings: [bs, 2, 256], dense embeddings)
                # The first emb in sparse_embeddings is torch.empty 
                object_bbox_emb = self.sam_prompt_encoder(boxes = torch.Tensor(self.sam_transform.apply_boxes(object_bbox, image_size)), points = None, masks = None)[0][0,0]
                subject_bbox_emb = self.sam_prompt_encoder(boxes = torch.Tensor(self.sam_transform.apply_boxes(subject_bbox, image_size)), points = None, masks = None)[0][0,0]
                predicate_emb = self.t5_encoder(input_ids=self.t5_tokenizer(relationship['predicate'], return_tensors="pt").input_ids).last_hidden_state[0,0]

                self.data.append({
                    'image_id': row['image_id'],
                    'image_path': os.path.join(self.image_folder, str(row['image_id']) + '.jpg'),
                    'predicate': relationship['predicate'],
                    'predicate_emb': predicate_emb.detach().numpy(),
                    'object': relationship['object'],
                    'subject': relationship['subject'],
                    'object_bbox_sam_emb': object_bbox_emb.detach().numpy(),
                    'subject_bbox_sam_emb': subject_bbox_emb.detach().numpy(),
                })
        self.data = pd.DataFrame(self.data)
        self.data.to_json(os.path.join(self.root_dir, 'relationships_dataset_with_emb.json'))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['image_path'])
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.Tensor(row['object_bbox_sam_emb']), torch.Tensor(row['subject_bbox_sam_emb']), torch.Tensor(row['predicate_emb'])


if __name__=='__main__':
    sam = sam_model_registry['vit_b'](checkpoint='/cmlscratch/nehamk/dense-image-representations/dense-image-representations/sam_vit_b_01ec64.pth')
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    encoder = T5EncoderModel.from_pretrained("t5-small")
    dataset = VisualGenomeRelationshipWithSAMAndT5(root_dir='/cmlscratch/nehamk/datasets/visual_genome', sam_model=sam, t5_tokenizer=tokenizer, t5_encoder=encoder)