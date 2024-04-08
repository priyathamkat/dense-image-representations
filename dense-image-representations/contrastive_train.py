import argparse
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel
import wandb
from losses import ContrastiveLoss
from modules import ProjectionHead


def train(
    image_encoder: TransformerEncoder,
    image_embeddings_projector: ProjectionHead,
    text_encoder: T5EncoderModel,
    text_embeddings_projector: ProjectionHead,
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

        image_encoder.train()
        image_embeddings_projector.train()
        text_encoder.train()
        text_embeddings_projector.train()

        for text_tokens, visual_tokens in train_dataloader:
            text_tokens = text_tokens.to('cuda')
            visual_tokens = visual_tokens.to('cuda')

            text_embeddings = text_encoder(text_tokens)
            text_embeddings = text_embeddings_projector(text_embeddings)

            image_embeddings = image_encoder(visual_tokens)
            image_embeddings = image_embeddings_projector(image_embeddings)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

            epoch_loss += loss.item() * text_tokens.shape[0]
        
        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss, "learning_rate": optimizer.param_groups[0]['lr']})

        if epoch % args.validation_epochs == 0:
            evaluate(image_encoder, image_embeddings_projector, text_encoder, text_embeddings_projector, val_dataloader, contrastive_loss)


def evaluate(
    image_encoder: TransformerEncoder,
    image_embeddings_projector: ProjectionHead,
    text_encoder: T5EncoderModel,
    text_embeddings_projector: ProjectionHead,
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

    image_encoder.eval()
    image_embeddings_projector.eval()
    text_encoder.eval()
    text_embeddings_projector.eval()

    loss = 0
    for text_tokens, visual_tokens in val_dataloader:
        text_tokens = text_tokens.to('cuda')
        visual_tokens = visual_tokens.to('cuda')

        with torch.no_grad():
            text_embeddings = text_encoder(text_tokens)
            text_embeddings = text_embeddings_projector(text_embeddings)

            image_embeddings = image_encoder(visual_tokens)
            image_embeddings = image_embeddings_projector(image_embeddings)

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * text_tokens.shape[0]
        
        # TODO: Add other metrics

    loss /= len(val_dataloader.dataset)

    wandb.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Evaluation of Generated T2I CompBench color dataset")

    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--vision_graph_data', type=str, required=True)
    parser.add_argument('--text_graph_data', type=str, required=True)

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

    image_encoder = TransformerEncoder(
        encoder_layer=TransformerEncoderLayer(d_model=512, nhead=8),
        num_layers=4
    )
    image_embeddings_projector = ProjectionHead(embedding_dim=1024, projection_dim=512, dropout=0.1) # TODO: Change the parameters
    
    tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    text_encoder = T5EncoderModel.from_pretrained("google-t5/t5-small")
    text_embeddings_projector = ProjectionHead(embedding_dim=768, projection_dim=512, dropout=0.1) # TODO: Change the parameters

    optimizer = torch.optim.Adam(
        list(image_encoder.parameters()) + list(image_embeddings_projector.parameters()) + list(text_encoder.parameters()) + list(text_embeddings_projector.parameters()),
        lr=args.lr
    )

    dataset = None # TODO: return the input to the tokenizer and the input to the image encoder
    train_dataloader = None # TODO
    val_dataloader = None # TODO

    init_wandb(args)

    train(
        image_encoder,
        image_embeddings_projector,
        tokenizer,
        text_encoder,
        text_embeddings_projector,
        optimizer,
        train_dataloader,
        val_dataloader,
        ContrastiveLoss(),
        args,
    )


if __name__ == "__main__":
    main()
