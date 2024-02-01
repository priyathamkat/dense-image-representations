import argparse
import torch 
from torch_geometric.loader import DataLoader

from data.graph_dataset import GraphDataset
from gcn.deeper_gcn import DeeperGCN

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--vision_graph_data', type=str)
parser.add_argument('--text_graph_data', type=str)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--hidden_channels', type=int, default=256)
parser.add_argument('--num_layers', type=int, default=28)

args = parser.parse_args()

vision_dataset = GraphDataset(args.vision_graph_data)
text_dataset = GraphDataset(args.text_graph_data)

vision_dataloader = DataLoader(
    vision_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

vision_encoder = DeeperGCN(
    in_channels = vision_dataset.num_features,
    hidden_channels = args.hidden_channels,
    num_layers = args.num_layers
)

for (i, batch) in enumerate(vision_dataloader):
    pdb.set_trace()
