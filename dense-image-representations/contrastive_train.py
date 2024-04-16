import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import wandb
from losses import ContrastiveLoss
from modules import VisionLanguageEncoder

from data.tokens import VisualAndTextTokens

import pdb

def train(
    vision_language_encoder: VisionLanguageEncoder,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader, 
    contrastive_loss: ContrastiveLoss,
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

        for batch in train_dataloader:
            text_tokens = batch[2].to('cuda')
            image_tokens = batch[0].to('cuda')
            image_attention_mask = batch[1].to('cuda')
            
            image_attention_mask[image_attention_mask == 0] = float('-inf')
            image_attention_mask[image_attention_mask == 1] = 0
            
            image_embeddings, text_embeddings = vision_language_encoder(image_tokens, image_attention_mask, text_tokens)

            image_embeddings = image_embeddings.mean(dim=1)
            text_embeddings = text_embeddings.mean(dim=1)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            epoch_loss += loss.item() * text_tokens.shape[0]
        
        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})

        if epoch % args.validation_epochs == 0:
            evaluate(vision_language_encoder, val_dataloader, contrastive_loss)


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
    for batch in val_dataloader:
        text_tokens = batch[2].to('cuda')
        image_tokens = batch[0].to('cuda')
        image_attention_mask = batch[1].to('cuda')

        with torch.no_grad():
            image_embeddings, text_embeddings = vision_language_encoder(image_tokens, image_attention_mask, text_tokens)

            image_embeddings = image_embeddings.mean(dim=1)
            text_embeddings = text_embeddings.mean(dim=1)

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * text_tokens.shape[0]
        
        # TODO: Add other metrics

    loss /= len(val_dataloader.dataset)

    wandb.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Evaluation of Generated T2I CompBench color dataset")

    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--vision_tokens', type=str, required=True)
    parser.add_argument('--text_tokens', type=str, required=True)

    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=7)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=16)

    parser.add_argument("--validation_epochs", type=int, default=10)

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

    vision_language_encoder = VisionLanguageEncoder(embed_dim=512, projection_dim=128, transformer_width=512, transformer_heads=8, transformer_layers=6)
    vision_language_encoder = vision_language_encoder.cuda()

    optimizer = torch.optim.Adam(
        vision_language_encoder.parameters(),
        lr=args.lr
    )

    dataset = VisualAndTextTokens(image_root=args.vision_tokens, text_root=args.text_tokens)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    init_wandb(args)

    train(
        vision_language_encoder,
        optimizer,
        train_dataloader,
        val_dataloader,
        ContrastiveLoss(),
        args,
    )


if __name__ == "__main__":
    main()
