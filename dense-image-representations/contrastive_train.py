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
import pdb

def get_avg_sim(sim_matrix):
    eye = torch.eye(sim_matrix.shape[0], device = sim_matrix.device).bool()
    diag = (sim_matrix * eye).nonzero()
    off_diag = (sim_matrix * ~eye).nonzero()
    return sim_matrix[diag[:,0], diag[:,1]].mean().item(), sim_matrix[off_diag[:,0], off_diag[:,1]].mean().item()

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

    for epoch in range(args.epochs):
        epoch_loss = 0

        vision_language_encoder.train()

        for i, batch in enumerate(train_dataloader):
            text_tokens = batch[2].to('cuda')
            image_tokens = batch[0].to('cuda')
            image_attention_mask = batch[1].to('cuda')
            
            # image_attention_mask[image_attention_mask == 0] = float('-inf')
            # image_attention_mask[image_attention_mask == 1] = 0

            image_attention_mask = torch.zeros(image_tokens.shape[0], image_tokens.shape[1], image_tokens.shape[1]).to('cuda')

            image_embeddings, text_embeddings = vision_language_encoder(image_tokens, image_attention_mask, text_tokens)

            image_embeddings = image_embeddings.mean(dim=1)
            text_embeddings = text_embeddings.mean(dim=1)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (len(train_dataloader.dataset) // args.batch_size) * epoch + i
            lr_scheduler(step)

            wandb.log({"loss": loss.item()})
            
            epoch_loss += loss.item() * text_tokens.shape[0]

        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})

        if epoch % args.validation_epochs == 0:
            evaluate(vision_language_encoder, val_dataloader, contrastive_loss)

        if epoch % args.checkpoint_epochs == 0:
            torch.save(vision_language_encoder.state_dict(), f"{checkpoint_dir}/model_{epoch}.pt")


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
        text_tokens = batch[2].to('cuda')
        image_tokens = batch[0].to('cuda')
        image_attention_mask = batch[1].to('cuda')

        with torch.no_grad():
            image_embeddings, text_embeddings = vision_language_encoder(image_tokens, image_attention_mask, text_tokens)

            image_embeddings = image_embeddings.mean(dim=1)
            text_embeddings = text_embeddings.mean(dim=1)

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * text_tokens.shape[0]
        
       
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

    loss /= len(val_dataloader.dataset)

    wandb.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Evaluation of Generated T2I CompBench color dataset")

    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--vision_tokens', type=str, required=True)
    parser.add_argument('--text_tokens', type=str, required=True)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--projection_dim', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_epochs", type=int, default=20)

    args = parser.parse_args()

    return args


def init_wandb(args):
    wandb.login()
    wandb.init(
        name = args.exp_name,
        project="graph-clip",
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
        },
    )


def main():
    args = parse_args()

    vision_language_encoder = VisionLanguageEncoder(embed_dim=512,
                                                    projection_dim=args.projection_dim, 
                                                    transformer_width=512, 
                                                    transformer_heads=args.num_heads, 
                                                    transformer_layers=args.num_layers)

    vision_language_encoder = vision_language_encoder.cuda()

    dataset = VisualAndTextTokens(image_root=args.vision_tokens, text_root=args.text_tokens)
    val_dataset = VisualAndTextTokens(image_root=args.vision_tokens + '_test', text_root=args.text_tokens + '_test')
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
    lr_scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    init_wandb(args)

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
