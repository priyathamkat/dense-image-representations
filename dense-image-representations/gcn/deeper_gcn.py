import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GENConv, DeepGCNLayer

import pdb

class DeeperGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return x


# class Encoder(torch.nn.Module):
#     def __init__(self, encoder, augmentor):
#         super(Encoder, self).__init__()
#         self.encoder = encoder
#         self.augmentor = augmentor

#     def forward(self, x, edge_index, batch):
#         aug1, aug2 = self.augmentor
#         x1, edge_index1, edge_weight1 = aug1(x, edge_index)
#         x2, edge_index2, edge_weight2 = aug2(x, edge_index)
#         z, g = self.encoder(x, edge_index, batch)
#         z1, g1 = self.encoder(x1, edge_index1, batch)
#         z2, g2 = self.encoder(x2, edge_index2, batch)
#         return z, g, z1, z2, g1, g2


# def train(encoder_model, contrast_model, dataloader, optimizer):
#     encoder_model.train()
#     epoch_loss = 0
#     for data in dataloader:
#         data = data.to('cuda')
#         optimizer.zero_grad()

#         if data.x is None:
#             num_nodes = data.batch.size(0)
#             data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

#         _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
#         g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
#         loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
#         loss.backward()
#         optimizer.step()

#         epoch_loss += loss.item()
#     return epoch_loss


# def test(encoder_model, dataloader):
#     encoder_model.eval()
#     x = []
#     y = []
#     for data in dataloader:
#         data = data.to('cuda')
#         if data.x is None:
#             num_nodes = data.batch.size(0)
#             data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
#         _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
#         x.append(g)
#         y.append(data.y)
#     x = torch.cat(x, dim=0)
#     y = torch.cat(y, dim=0)

#     split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
#     result = SVMEvaluator(linear=True)(x, y, split)
#     return result

def main():
    device = torch.device('cuda')
    
    input_dim = 512

    gconv = DeeperGCN(in_channels = input_dim, hidden_channels = 64, num_layers = 28).to(device)
    gr = torch.load('../coco_visual_graph/30_graph.pt').to(device)
    out = gconv(gr.x, gr.edge_index, gr.edge_attr) # nodes x hidden_channels


if __name__ == '__main__':
    main()