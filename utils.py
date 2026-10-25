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
