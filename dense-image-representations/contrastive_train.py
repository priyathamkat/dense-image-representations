import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from losses import ContrastiveLoss
from modules import VisionLanguageEncoder

from data.datautils import get_dataset
import utils

import os
import glob
import pdb
import clip

def forward_pass(vision_language_encoder, batch):
    image_tokens = batch['image_tokens'].cuda()
    image_features = batch['image_features'].cuda()
    num_non_pad_tokens = batch['num_non_pad_tokens'].cuda()
    num_nodes = batch['num_nodes'].cuda()
    # image_attention_mask = batch[4].cuda()
    text_tokens = batch['captions'].cuda()
    
    # image_attention_mask = image_attention_mask.float().masked_fill(image_attention_mask == 0, float('-inf')).masked_fill(image_attention_mask == 1, float(0.0))
    # image_attention_mask = ~(image_attention_mask.bool())
    # image_attention_mask = torch.zeros(image_tokens.shape[0], image_tokens.shape[1], image_tokens.shape[1]).to('cuda').bool()

    image_embeddings, text_embeddings = vision_language_encoder(image_tokens,
                                                                image_features,
                                                                num_non_pad_tokens,
                                                                num_nodes,
                                                                text_tokens,
                                                                )
    if len(image_embeddings.shape) == 3:
        image_embeddings = image_embeddings.mean(dim=1)
    if len(text_embeddings.shape) == 3:
        text_embeddings = text_embeddings.mean(dim=1)

    assert image_embeddings.shape[1] == text_embeddings.shape[1]

    return image_embeddings, text_embeddings

def train(
    vision_language_encoder: VisionLanguageEncoder,
    tokenizer,
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
            # tokenize
            batch['captions'] = utils.tokenize(batch['captions'], tokenizer, args.text_encoder)

            image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (len(train_dataloader.dataset) // args.batch_size) * epoch + i
            lr_scheduler(step)

            wandb.log({"loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch})

            epoch_loss += loss.item() * batch['captions'].shape[0]

        epoch_loss /= len(train_dataloader.dataset)

        wandb.log({"epoch_training_loss": loss})

        if epoch % args.validation_epochs == 0:
            evaluate(vision_language_encoder, tokenizer, val_dataloader, contrastive_loss, args)

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
    tokenizer,
    val_dataloader: DataLoader,
    contrastive_loss: ContrastiveLoss, 
    args: argparse.Namespace,
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
            # tokenize
            batch['captions'] = utils.tokenize(batch['captions'], tokenizer, args.text_encoder)

            image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

            loss += contrastive_loss(text_embeddings, image_embeddings).item() * batch['captions'].shape[0]
        
       
            x1.append(image_embeddings)
            x2.append(text_embeddings)

    x1 = torch.cat(x1, dim=0)
    x2 = torch.cat(x2, dim=0)

    x1 = F.normalize(x1, dim = 1)
    x2 = F.normalize(x2, dim = 1)
    
    sim_1_1 = torch.matmul(x1, x1.T)
    sim_2_2 = torch.matmul(x2, x2.T)
    sim_1_2 = torch.matmul(x1, x2.T)
    
    diag_sim_v_v, off_diag_sim_v_v = utils.get_avg_sim(sim_1_1)
    wandb.log({"diag_sim_v_v": diag_sim_v_v, "off_diag_sim_v_v": off_diag_sim_v_v})

    diag_sim_t_t, off_diag_sim_t_t = utils.get_avg_sim(sim_2_2)
    wandb.log({"diag_sim_t_t": diag_sim_t_t, "off_diag_sim_t_t": off_diag_sim_t_t})

    diag_sim_v_t, off_diag_sim_v_t = utils.get_avg_sim(sim_1_2)
    wandb.log({"diag_sim_v_t": diag_sim_v_t, "off_diag_sim_v_t": off_diag_sim_v_t})   

    utils.get_retrieval_score(sim_1_2, log_name='v_t')
    sim_2_1 = sim_1_2.T
    utils.get_retrieval_score(sim_2_1, log_name='t_v')
    
    loss /= len(val_dataloader.dataset)

    wandb.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(description="VQA Evaluation of Generated T2I CompBench color dataset")

    parser.add_argument('--exp_name', type=str, required=True)

    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--projection_dim', type=int, default=512)
    parser.add_argument('--preembed_nodes', action='store_true')
    parser.add_argument('--text_encoder', type=str, default='t5_small')

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

    clip_model = None
    if args.text_encoder == 'clip':
        clip_model, _ = clip.load("ViT-B/32", device='cuda')
        clip_model = clip_model.to(torch.float32)

    vision_language_encoder = VisionLanguageEncoder(projection_dim=args.projection_dim,
                                                    transformer_width=512, 
                                                    transformer_heads=args.num_heads, 
                                                    transformer_layers=args.num_layers,
                                                    image_embedding_size=2880,
                                                    preembed_nodes=args.preembed_nodes,
                                                    text_encoder=args.text_encoder,
                                                    clip_model=clip_model,)

    vision_language_encoder = vision_language_encoder.cuda()

    tokenizer = utils.get_tokenizer(args.text_encoder)

    dataset = get_dataset(
        dataset_name = args.dataset,
        image_tokens_root = f'{args.dataset}_visual_tokens',
        with_image_tokens = True, 
        caption_return_policy = 'random'
    )
    val_dataset = get_dataset(
        dataset_name = args.dataset + '_val',
        image_tokens_root = f'{args.dataset}_val_visual_tokens',
        with_image_tokens = True, 
        caption_return_policy = 'random'
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

    optimizer = torch.optim.AdamW(
        vision_language_encoder.parameters(),
        lr = args.lr,
        betas = (args.beta1, args.beta2),
        eps = args.eps,
    )

    total_steps = (len(dataset) // args.batch_size) * args.epochs
    warmup_steps = (len(dataset) // args.batch_size) * args.warmup 
    lr_scheduler = utils.cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    init_wandb(args)
    wandb.watch(vision_language_encoder, log="all")

    checkpoint_dir = f"results/{args.exp_name}"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train(
        vision_language_encoder,
        tokenizer,
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
