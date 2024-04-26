import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
from losses import ContrastiveLoss
from modules import VisionLanguageEncoder

from data.tokens import VisualAndTextTokens

import os
import numpy as np
import glob
import pdb

def get_avg_sim(sim_matrix):
    eye = torch.eye(sim_matrix.shape[0], device = sim_matrix.device).bool()
    diag = (sim_matrix * eye).nonzero()
    off_diag = (sim_matrix * ~eye).nonzero()
    return sim_matrix[diag[:,0], diag[:,1]].mean().item(), sim_matrix[off_diag[:,0], off_diag[:,1]].mean().item()

def get_retrieval_score(sim_1_2, log_name='v_t'):
    ranked_sim = sim_1_2.sort(descending=True).indices
    ranks = []
    for i in range(ranked_sim.shape[0]):
        ranks.append(torch.where(ranked_sim[i] == i)[0][0].item())
    
    ranks = torch.Tensor(ranks)
    r1 = 100.0 * len(torch.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(torch.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(torch.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(torch.where(ranks < 50)[0]) / len(ranks)
    medr = torch.floor(torch.median(ranks)) + 1
    meanr = ranks.mean() + 1
    
    wandb.log({f"{log_name}_r1": r1, f"{log_name}_r5": r5, f"{log_name}_r10": r10, f"{log_name}_r50": r50, f"{log_name}_medr": medr, f"{log_name}_meanr": meanr})


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def forward_pass(vision_language_encoder, batch):
    image_tokens = batch[0].cuda()
    image_features = batch[1].cuda()
    num_non_pad_tokens = batch[2].cuda()
    num_nodes = batch[3].cuda()
    # image_attention_mask = batch[4].cuda()
    text_tokens = batch[5].cuda()
    
    # image_attention_mask = image_attention_mask.float().masked_fill(image_attention_mask == 0, float('-inf')).masked_fill(image_attention_mask == 1, float(0.0))
    # image_attention_mask = ~(image_attention_mask.bool())
    # image_attention_mask = torch.zeros(image_tokens.shape[0], image_tokens.shape[1], image_tokens.shape[1]).to('cuda').bool()

    image_embeddings, text_embeddings = vision_language_encoder(image_tokens,
                                                                image_features,
                                                                num_non_pad_tokens,
                                                                num_nodes,
                                                                text_tokens,
                                                                )

    image_embeddings = image_embeddings.mean(dim=1)
    text_embeddings = text_embeddings.mean(dim=1)

    return image_embeddings, text_embeddings

def train(
    vision_language_encoder: VisionLanguageEncoder,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader, 
    contrastive_loss: ContrastiveLoss,
    checkpoint_dir: str,
    args: argparse.Namespace,
):
    """
    Trains the image and text encoders (and the projection heads) using contrastive learning.

    Args:
        image_encoder (TransformerEncoder): The image encoder model.
        image_embeddings_projector (ProjectionHead): The projector for image embeddings.
        text_encoder (T5EncoderModel): The text encoder model.
        text_embeddings_projector (ProjectionHead): The projector for text embeddings.
        optimizer (torch.optim.Optimizer): The optimizer for updating models parameters.
        train_dataloader (DataLoader): The dataloader for training data.
        val_dataloader (DataLoader): The dataloader for validation data.
        contrastive_loss (ContrastiveLoss): The contrastive loss function.
        args (argparse.Namespace): The arguments.

    Returns:
        None
    """

    start_epoch = 0
    # load checkpoint if exists
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/model_*.pth.tar"), key=os.path.getmtime, reverse=True)
    if len(ckpts) > 0:
        print(f"Loading state dict {ckpts[0]}")
        state = torch.load(ckpts[0])
        vision_language_encoder.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer_state'])
        start_epoch = state['epoch'] + 1

    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0

        vision_language_encoder.train()

        for i, batch in enumerate(train_dataloader):
            image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (len(train_dataloader.dataset) // args.batch_size) * epoch + i
            lr_scheduler(step)

            wandb.log({"loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch})

            epoch_loss += loss.item() * batch[0].shape[0]

        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss})

        if epoch % args.validation_epochs == 0:
            evaluate(vision_language_encoder, val_dataloader, contrastive_loss)

        if epoch % args.checkpoint_epochs == 0 or epoch == args.epochs - 1:
            state = {'state_dict': vision_language_encoder.state_dict(), 
                    'epoch': epoch,
                    'optimizer_state': optimizer.state_dict()}
            ckpts = glob.glob(f"{checkpoint_dir}/model_*.pt*")
            for ckpt in ckpts:
                os.remove(ckpt)
            torch.save(state, f"{checkpoint_dir}/model_{epoch}.pth.tar")


def evaluate(
    vision_language_encoder: VisionLanguageEncoder,
    val_dataloader: DataLoader,
    contrastive_loss: ContrastiveLoss, 
):
    """
    Evaluate the performance of the models on the validation dataset.

    Args:
        image_encoder (TransformerEncoder): The image encoder model.
        image_embeddings_projector (ProjectionHead): The projection head for image embeddings.
        text_encoder (T5EncoderModel): The text encoder model.
        text_embeddings_projector (ProjectionHead): The projection head for text embeddings.
        val_dataloader (DataLoader): The validation dataloader.
        contrastive_loss (ContrastiveLoss): The contrastive loss function.

    Returns:
        None
    """

    vision_language_encoder.eval()

    loss = 0
    x1 = []
    x2 = []
    for batch in val_dataloader:
        with torch.no_grad():
            image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * batch[0].shape[0]
        
       
            x1.append(image_embeddings)
            x2.append(text_embeddings)

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

    get_retrieval_score(sim_1_2, log_name='v_t')
    sim_2_1 = sim_1_2.T
    get_retrieval_score(sim_2_1, log_name='t_v')
    
    loss /= len(val_dataloader.dataset)

    wandb.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Evaluation of Generated T2I CompBench color dataset")

    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--vision_tokens_train', type=str, required=True)
    parser.add_argument('--vision_tokens_val', type=str, required=True)
    parser.add_argument('--text_tokens_train', type=str, required=True)
    parser.add_argument('--text_tokens_val', type=str, required=True)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--preembed_nodes', action='store_true')

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_epochs", type=int, default=50)

    args = parser.parse_args()

    return args


def init_wandb(args):
    wandb.login()
    wandb.init(
        name = args.exp_name,
        project="graph-clip",
        config=args,
    )


def main():
    args = parse_args()

    vision_language_encoder = VisionLanguageEncoder(embed_dim=512,
                                                    projection_dim=args.projection_dim, 
                                                    transformer_width=512, 
                                                    transformer_heads=args.num_heads, 
                                                    transformer_layers=args.num_layers,
                                                    image_embedding_size=2880,
                                                    preembed_nodes=args.preembed_nodes,)

    vision_language_encoder = vision_language_encoder.cuda()

    dataset = VisualAndTextTokens(image_root=args.vision_tokens_train, text_root=args.text_tokens_train)
    val_dataset = VisualAndTextTokens(image_root=args.vision_tokens_val, text_root=args.text_tokens_val)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    optimizer = torch.optim.AdamW(
        vision_language_encoder.parameters(),
        lr = args.lr,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
    )

    total_steps = (len(dataset) // args.batch_size) * args.epochs
    warmup_steps = (len(dataset) // args.batch_size) * args.warmup 
    lr_scheduler = cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    init_wandb(args)
    wandb.watch(vision_language_encoder, log="all")

    checkpoint_dir = f"results/{args.exp_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(
        vision_language_encoder,
        optimizer,
        lr_scheduler,
        train_dataloader,
        val_dataloader,
        ContrastiveLoss(),
        checkpoint_dir,
        args,
    )


if __name__ == "__main__":
    main()
