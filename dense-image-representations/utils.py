import torch
import wandb 
import clip
import numpy as np
from transformers import AutoTokenizer

def get_tokenizer(text_encoder_type = 'clip'):
    tokenizer = None
    if text_encoder_type == 't5_small':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
    elif text_encoder_type == 't5_base':
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
    return tokenizer

def tokenize(captions, tokenizer = None, text_encoder_type = 'clip'):
    if 't5' in text_encoder_type:
        tokens = tokenizer(captions, return_tensors="pt", padding='max_length', max_length=77, truncation=True).input_ids.cpu()
    else:
        tokens = clip.tokenize(captions).cpu()
    return tokens


def get_avg_sim(sim_matrix):
    eye = torch.eye(sim_matrix.shape[0], device = sim_matrix.device).bool()
    diag = (sim_matrix * eye).nonzero()
    off_diag = (sim_matrix * ~eye).nonzero()
    return sim_matrix[diag[:,0], diag[:,1]].mean().item(), sim_matrix[off_diag[:,0], off_diag[:,1]].mean().item()

def get_retrieval_score(sim_1_2, accelerator, log_name='v_t'):
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
    
    accelerator.log({f"{log_name}_r1": r1, f"{log_name}_r5": r5, f"{log_name}_r10": r10, f"{log_name}_r50": r50, f"{log_name}_medr": medr, f"{log_name}_meanr": meanr})


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
