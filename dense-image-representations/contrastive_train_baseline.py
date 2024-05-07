import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as T
import wandb
from losses import ContrastiveLoss
from modules import VisionLanguageEncoderBase
from transformers import ViTImageProcessor

from data.coco import CocoImagesAndTextTokensForViT

import clip

import os
from PIL import Image
import numpy as np
import glob
import pdb

from contrastive_train import get_avg_sim, get_retrieval_score, assign_learning_rate, cosine_lr


def forward_pass(vision_language_encoder, batch):
    images = batch[0].to('cuda')
    text_tokens = batch[1].cuda()
    
    image_embeddings, text_embeddings = vision_language_encoder(images, text_tokens)

    if len(text_embeddings.shape) == 3:
        text_embeddings = text_embeddings.mean(dim=1)

    assert image_embeddings.shape[1] == text_embeddings.shape[1]

    return image_embeddings, text_embeddings

def train(
    vision_language_encoder: VisionLanguageEncoderBase,
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

            epoch_loss += loss.item() * batch[1].shape[0]

        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss})

        if epoch % args.validation_epochs == 0:
            evaluate(vision_language_encoder, val_dataloader, contrastive_loss)

        if epoch % args.checkpoint_epochs == 0 or epoch == args.epochs - 1:
            state = {
                'state_dict': vision_language_encoder.state_dict(), 
                'epoch': epoch,
                'optimizer_state': optimizer.state_dict()
            }
            ckpts = glob.glob(f"{checkpoint_dir}/model_*.pt*")
            for ckpt in ckpts:
                os.remove(ckpt)
            torch.save(state, f"{checkpoint_dir}/model_{epoch}.pth.tar")


def evaluate(
    vision_language_encoder,
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

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * batch[1].shape[0]
        
       
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

    parser.add_argument('--text_tokens_train', type=str, required=True)
    parser.add_argument('--text_tokens_val', type=str, required=True)

    parser.add_argument('--text_encoder', type=str, default='t5_small')
    parser.add_argument('--image_encoder', type=str, default='vit')
    parser.add_argument('--projection_dim', type=int, default=512)

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


def get_data_loaders(args, vit_processor):
    dataset = CocoImagesAndTextTokensForViT(
        image_root='/fs/cml-datasets/coco/images/train2017/',
        image_annFile='/fs/cml-datasets/coco/annotations/captions_train2017.json',
        vit_processor=vit_processor,
        text_root=args.text_tokens_train,
        text_tokenizer_type=args.text_encoder
    )
    val_dataset = CocoImagesAndTextTokensForViT(
        image_root='/fs/cml-datasets/coco/images/val2017/',
        image_annFile='/fs/cml-datasets/coco/annotations/captions_val2017.json',
        vit_processor=vit_processor,
        text_root=args.text_tokens_val,
        text_tokenizer_type=args.text_encoder
    )

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

    return train_dataloader, val_dataloader


def main():
    args = parse_args()
    
    clip_model = None
    if 'clip' in [args.image_encoder, args.text_encoder]:
        clip_model, clip_image_processor = clip.load("ViT-B/16", device='cuda')
        clip_model = clip_model.to(torch.float32)

    vision_language_encoder = VisionLanguageEncoderBase(projection_dim=args.projection_dim,
                                                        text_encoder=args.text_encoder,
                                                        image_encoder=args.image_encoder,
                                                        clip_model=clip_model,)
    vision_language_encoder = vision_language_encoder.cuda()
    vision_language_encoder = nn.DataParallel(vision_language_encoder)

    if args.image_encoder == 'vit':
        image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    else:
        image_processor = clip_image_processor

    train_dataloader, val_dataloader = get_data_loaders(args, image_processor)

    optimizer = torch.optim.AdamW(
        vision_language_encoder.parameters(),
        lr = args.lr,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
    )

    total_steps = (len(train_dataloader.dataset) // args.batch_size) * args.epochs
    warmup_steps = (len(train_dataloader.dataset) // args.batch_size) * args.warmup 
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
