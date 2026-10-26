"""Cross-platform model training helpers."""
import os
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.nn import functional as F

def read_config(path):
    with open(path,encoding="utf-8") as f: return yaml.safe_load(f)

def choose_device(preferred="auto"):
    if preferred=="auto":
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    if preferred=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if preferred=="mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS unavailable")
    return torch.device(preferred)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def bce_dice_loss(logits,target):
    bce=F.binary_cross_entropy_with_logits(logits,target)
    p=torch.sigmoid(logits)
    dims=(1,2,3)
    inter=(p*target).sum(dim=dims)
    denom=p.sum(dim=dims)+target.sum(dim=dims)
    return bce+1-((2*inter+1)/(denom+1)).mean()

def bce_tversky_loss(logits, target, fp_weight=0.3, fn_weight=0.7):
    """BCE plus a recall-weighted Tversky term for small foreground objects."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probability * target).sum(dim=dims)
    fp = (probability * (1 - target)).sum(dim=dims)
    fn = ((1 - probability) * target).sum(dim=dims)
    score = (tp + 1) / (tp + fp_weight * fp + fn_weight * fn + 1)
    return bce + 1 - score.mean()

@torch.no_grad()
def dice_iou_sum(logits,masks):
    pred=torch.sigmoid(logits)>0.5
    gt=masks>0.5
    dims=(1,2,3)
    inter=(pred&gt).sum(dim=dims).float()
    a=pred.sum(dim=dims).float()
    b=gt.sum(dim=dims).float()
    d=(2*inter+1e-6)/(a+b+1e-6)
    i=(inter+1e-6)/(a+b-inter+1e-6)
    return d.sum().item(),i.sum().item(),masks.shape[0]

@torch.no_grad()
def per_image_metrics(logits, masks, threshold=0.5):
    """Binary metrics per image; an empty prediction against a positive GT scores 0."""
    pred = torch.sigmoid(logits) > threshold
    gt = masks > 0.5
    dims = (1, 2, 3)
    tp = (pred & gt).sum(dim=dims).float()
    predicted = pred.sum(dim=dims).float()
    actual = gt.sum(dim=dims).float()
    union = predicted + actual - tp
    dice = torch.where(predicted + actual == 0, 1.0,
                       2 * tp / (predicted + actual).clamp_min(1))
    iou = torch.where(union == 0, 1.0, tp / union.clamp_min(1))
    precision = torch.where(predicted == 0, (actual == 0).float(),
                            tp / predicted.clamp_min(1))
    recall = torch.where(actual == 0, 1.0, tp / actual.clamp_min(1))
    foreground_fraction = predicted / pred[0].numel()
    values = zip(dice, iou, precision, recall, foreground_fraction)
    return [dict(zip(("dice", "iou", "precision", "recall", "foreground_fraction"),
                     (float(x) for x in row))) for row in values]

def move_optimizer_states(optimizer,device):
    for state in optimizer.state.values():
        for key,value in state.items():
            if isinstance(value,torch.Tensor) and key!="step":
                state[key]=value.to(device)

def atomic_torch_save(payload,destination):
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix(destination.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,destination)
