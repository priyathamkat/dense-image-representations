import argparse
import torch 
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.loader import DataLoader

import GCL.losses as L
from GCL.models import DualBranchContrast

from data.graph_dataset import ImageTextGraphDataset
from gcn.deeper_gcn import DeeperGCN

import wandb
import pdb

def train(vision_encoder_model, text_encoder_model, contrast_model, dataloader, optimizer):
    vision_encoder_model.train()
    text_encoder_model.train()
    epoch_loss = 0
    for i, batch in enumerate(dataloader):
        vision_data, text_data = batch
        vision_data = vision_data.to('cuda')
        text_data = text_data.to('cuda')
        optimizer.zero_grad()

        # if data.x is None:
        #     num_nodes = data.batch.size(0)
        #     data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z1, g1 = vision_encoder_model(vision_data.x, vision_data.edge_index, vision_data.edge_attr, vision_data.batch)
        z2, g2 = text_encoder_model(text_data.x, text_data.edge_index, text_data.edge_attr, text_data.batch)

        g1 = vision_encoder_model.project(g1)
        g2 = text_encoder_model.project(g2)

        loss = contrast_model(g1=g1, g2=g2, batch=vision_data.batch)
        loss += contrast_model(g1=g2, g2=g1, batch=text_data.batch)
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

        epoch_loss += loss.item()
    return epoch_loss


def get_avg_sim(sim_matrix):
    eye = torch.eye(sim_matrix.shape[0], device = sim_matrix.device).bool()
    diag = (sim_matrix * eye).nonzero()
    off_diag = (sim_matrix * ~eye).nonzero()
    return sim_matrix[diag[:,0], diag[:,1]].mean().item(), sim_matrix[off_diag[:,0], off_diag[:,1]].mean().item()


def test(vision_encoder_model, text_encoder_model, dataloader):
    vision_encoder_model.eval()
    text_encoder_model.eval()
    x1 = []
    x2 = []
    for i, batch in enumerate(dataloader):
        vision_data, text_data = batch
        vision_data = vision_data.to('cuda')
        text_data = text_data.to('cuda')
        optimizer.zero_grad()

        _, g1 = vision_encoder_model(vision_data.x, vision_data.edge_index, vision_data.edge_attr, vision_data.batch)
        _, g2 = text_encoder_model(text_data.x, text_data.edge_index, text_data.edge_attr, text_data.batch)

        x1.append(g1)
        x2.append(g2)

    x1 = torch.cat(x1, dim=0)
    x2 = torch.cat(x2, dim=0)

    x1 = F.normalize(x1, dim = 1)
    x2 = F.normalize(x2, dim = 1)
    
    sim_1_1 = torch.matmul(x1, x1.T)
    sim_2_2 = torch.matmul(x2, x2.T)
    sim_1_2 = torch.matmul(x1, x2.T)
    
    diag_sim_v_v, off_diag_sim_v_v = get_avg_sim(sim_1_1)
    wandb.log({"diag_sim_v_v": diag_sim_v_v, "off_diag_sim_v_v": off_diag_sim_v_v})

    diag_sim_t_t, off_diag_sim_t_t = get_avg_sim(sim_2_2)
    wandb.log({"diag_sim_t_t": diag_sim_t_t, "off_diag_sim_t_t": off_diag_sim_t_t})

    diag_sim_v_t, off_diag_sim_v_t = get_avg_sim(sim_1_2)
    wandb.log({"diag_sim_v_t": diag_sim_v_t, "off_diag_sim_v_t": off_diag_sim_v_t})


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str)

parser.add_argument('--vision_graph_data', type=str)
parser.add_argument('--text_graph_data', type=str)

parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=7)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

device = torch.device('cuda')

graph_dataset = ImageTextGraphDataset(image_root=args.vision_graph_data, text_root=args.text_graph_data)
test_dataset = ImageTextGraphDataset(image_root=f'{args.vision_graph_data}_test', text_root=f'{args.text_graph_data}_test')

graph_dataloader = DataLoader(
    graph_dataset,
    batch_size=args.batch_size,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False
)

vision_encoder = DeeperGCN(
    in_channels = graph_dataset.num_features,
    hidden_channels = args.hidden_channels,
    num_layers = args.num_layers
).to(device)


text_encoder = DeeperGCN(
    in_channels = graph_dataset.num_features,
    hidden_channels = args.hidden_channels,
    num_layers = args.num_layers
).to(device)

contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(device)
optimizer = Adam(list(vision_encoder.parameters()) + list(text_encoder.parameters()), lr=args.lr)


wandb.login()
wandb.init(
    name = args.exp_name,
    project="graph-clip",
    config={
        "learning_rate": args.lr,
        "epochs": args.epochs,
    },
)


for epoch in range(args.epochs):
    loss = train(vision_encoder, text_encoder, contrast_model, graph_dataloader, optimizer)
    wandb.log({"epoch_loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})
        
    if epoch % 10 == 0:
        test(vision_encoder, text_encoder, test_dataloader)

