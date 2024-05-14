import argparse
import glob
import os
import pdb
import shutil
import re

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import wandb
from accelerate import Accelerator
from data.datautils import get_dataset
from losses import ContrastiveLoss
from modules import VisionLanguageEncoderBase
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor
from accelerate.utils import DistributedDataParallelKwargs


def forward_pass(vision_language_encoder, batch):
    images = batch["images"]
    text_tokens = batch["captions"]

    image_embeddings, text_embeddings = vision_language_encoder(images, text_tokens)

    if len(text_embeddings.shape) == 3:
        text_embeddings = text_embeddings.mean(dim=1)

    assert image_embeddings.shape[1] == text_embeddings.shape[1]

    return image_embeddings, text_embeddings


def train(
    vision_language_encoder: VisionLanguageEncoderBase,
    tokenizer,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    contrastive_loss: ContrastiveLoss,
    checkpoint_dir: str,
    args: argparse.Namespace,
    accelerator: Accelerator,
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
    ckpts = sorted(glob.glob(f"{checkpoint_dir}/model_*"), key=os.path.getmtime, reverse=True)
    if len(ckpts) > 0:
        accelerator.wait_for_everyone()
        print(f"Loading state {ckpts[0]}")
        start_epoch = int(re.findall(r'model_(\d+)', ckpts[0])[0]) + 1
        accelerator.load_state(input_dir=checkpoint_dir)

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0

        vision_language_encoder.train()

        for i, batch in enumerate(train_dataloader):
            # tokenize
            batch["captions"] = utils.tokenize(batch["captions"], tokenizer, args.text_encoder)

            image_embeddings, text_embeddings = forward_pass(vision_language_encoder, batch)

            loss = contrastive_loss(text_embeddings, image_embeddings)

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            step = (len(train_dataloader.dataset) // args.batch_size) * epoch + i
            lr_scheduler(step)

            accelerator.log(
                {
                    "loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
            )

            epoch_loss += loss.item() * batch["captions"].shape[0]

        epoch_loss /= len(train_dataloader.dataset)

        accelerator.log({"epoch_training_loss": epoch_loss})

        if epoch % args.validation_epochs == 0:
            evaluate(
                vision_language_encoder,
                tokenizer,
                val_dataloader,
                contrastive_loss,
                args,
                accelerator,
            )

        if epoch % args.checkpoint_epochs == 0 or epoch == args.epochs - 1:
            accelerator.wait_for_everyone()

            ckpts = glob.glob(f"{checkpoint_dir}/model_*")
            for ckpt in ckpts:
                shutil.rmtree(ckpt, ignore_errors=True)
            
            accelerator.save_state(output_dir=f"{checkpoint_dir}")
            accelerator.save_model(vision_language_encoder, f"{checkpoint_dir}/model_{epoch}")

@torch.no_grad()
def evaluate(
    vision_language_encoder,
    tokenizer,
    val_dataloader: DataLoader,
    contrastive_loss: ContrastiveLoss,
    args: argparse.Namespace,
    accelerator: Accelerator,
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
            batch["captions"] = utils.tokenize(
                batch["captions"], tokenizer, args.text_encoder
            )

            image_embeddings, text_embeddings = forward_pass(
                vision_language_encoder, batch
            )

            loss += (
                contrastive_loss(text_embeddings, image_embeddings).item()
                * batch["captions"].shape[0]
            )

            x1.append(image_embeddings)
            x2.append(text_embeddings)

    x1 = torch.cat(x1, dim=0)
    x2 = torch.cat(x2, dim=0)

    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    sim_1_1 = torch.matmul(x1, x1.T)
    sim_2_2 = torch.matmul(x2, x2.T)
    sim_1_2 = torch.matmul(x1, x2.T)

    diag_sim_v_v, off_diag_sim_v_v = utils.get_avg_sim(sim_1_1)
    accelerator.log(
        {"diag_sim_v_v": diag_sim_v_v, "off_diag_sim_v_v": off_diag_sim_v_v}
    )

    diag_sim_t_t, off_diag_sim_t_t = utils.get_avg_sim(sim_2_2)
    accelerator.log(
        {"diag_sim_t_t": diag_sim_t_t, "off_diag_sim_t_t": off_diag_sim_t_t}
    )

    diag_sim_v_t, off_diag_sim_v_t = utils.get_avg_sim(sim_1_2)
    accelerator.log(
        {"diag_sim_v_t": diag_sim_v_t, "off_diag_sim_v_t": off_diag_sim_v_t}
    )

    utils.get_retrieval_score(sim_1_2, log_name="v_t", accelerator=accelerator)
    sim_2_1 = sim_1_2.T
    utils.get_retrieval_score(sim_2_1, log_name="t_v", accelerator=accelerator)

    loss /= len(val_dataloader.dataset)

    accelerator.log({"validation_loss": loss})


def parse_args():
    parser = argparse.ArgumentParser(
        description="VQA Evaluation of Generated T2I CompBench color dataset"
    )

    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)

    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument("--text_encoder", type=str, default="t5_small")
    parser.add_argument("--image_encoder", type=str, default="vit")
    parser.add_argument("--projection_dim", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--validation_epochs", type=int, default=10)
    parser.add_argument("--checkpoint_epochs", type=int, default=50)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs])

    accelerator.init_trackers(project_name="graph-clip", init_kwargs={"wandb":{"name":args.exp_name}}, config=args)

    clip_model = None
    if "clip" in [args.image_encoder, args.text_encoder]:
        clip_model, clip_image_processor = clip.load("ViT-B/32")
        clip_model = clip_model.to(torch.float32)

    vision_language_encoder = VisionLanguageEncoderBase(
        projection_dim=args.projection_dim,
        text_encoder=args.text_encoder,
        image_encoder=args.image_encoder,
        clip_model=clip_model,
    )
    device = accelerator.device
    vision_language_encoder = vision_language_encoder.to(device)

    vision_language_encoder = accelerator.prepare(vision_language_encoder)

    if 'vit' in args.image_encoder:
        image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-224-in21k')
    else:
        image_processor = clip_image_processor

    tokenizer = utils.get_tokenizer(args.text_encoder)

    dataset = get_dataset(
        dataset_name=args.dataset,
        transform=image_processor,
        with_image_tokens=False,
        caption_return_policy="random",
        hf_vit_processor = 'vit' in args.image_encoder,
    )
    val_dataset = get_dataset(
        dataset_name=args.dataset + "_val",
        transform=image_processor,
        with_image_tokens=False,
        caption_return_policy="random",
        hf_vit_processor = 'vit' in args.image_encoder,
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
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    total_steps = (len(train_dataloader.dataset) // args.batch_size) * args.epochs
    warmup_steps = (len(train_dataloader.dataset) // args.batch_size) * args.warmup
    lr_scheduler = utils.cosine_lr(optimizer, args.lr, warmup_steps, total_steps)

    train_dataloader, val_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, val_dataloader, optimizer, lr_scheduler
    )

    checkpoint_dir = f"{args.result_dir}/{args.exp_name}"

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
        accelerator,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
