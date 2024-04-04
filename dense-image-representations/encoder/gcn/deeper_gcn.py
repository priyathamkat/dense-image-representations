import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GENConv, DeepGCNLayer, global_add_pool

import pdb

class DeeperGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        # for i in range(1, num_layers + 1):
        for i in range(num_layers):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        project_dim = hidden_channels * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, edge_attr, batch):
        z = x
        zs = []

        z = self.node_encoder(z)
        edge_attr = self.edge_encoder(edge_attr)

        # z = self.layers[0].conv(z, edge_index, edge_attr)
        # zs.append(z)

        for layer in self.layers[1:]:
            z = layer(z, edge_index, edge_attr)
            zs.append(z)

        # z = self.layers[0].act(self.layers[0].norm(z))
        # z = F.dropout(z, p=0.1, training=self.training)

        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g

def main():
    device = torch.device('cuda')
    
    input_dim = 512

    gconv = DeeperGCN(in_channels = input_dim, hidden_channels = 64, num_layers = 28).to(device)
    gr = torch.load('../coco_visual_graph/217205_0.pt').to(device)
    out = gconv(gr.x, gr.edge_index, gr.edge_attr) # nodes x hidden_channels
    pdb.set_trace()


if __name__ == '__main__':
    main()