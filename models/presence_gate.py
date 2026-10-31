"""Frozen feature pooling and a source-calibrated image-presence classifier."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def unet_features(model, images):
    """Same frozen U-Net forward; also return its bottleneck, without hooks."""
    x = images
    skips = []
    for encoder in model.encoders:
        x = encoder(x)
        skips.append(x)
        x = model.pool(x)
    bottleneck = model.bottleneck(x)
    x = bottleneck
    for decoder, skip in zip(model.decoders, reversed(skips)):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = decoder(torch.cat((skip, x), 1))
    return model.out(x), bottleneck


def pool_features(features, foreground, valid=None):
    features = features.float()
    foreground = F.interpolate(foreground.float(), size=features.shape[-2:],
                               mode="bilinear", align_corners=False).clamp(0, 1)
    if valid is None:
        valid = torch.ones_like(foreground)
    else:
        valid = F.interpolate(valid.float(), size=features.shape[-2:], mode="area")
    def weighted(mask):
        denominator = mask.sum((2, 3))
        mean = (features * mask).sum((2, 3)) / denominator.clamp_min(1e-6)
        return mean, denominator
    global_mean, _ = weighted(valid)
    global_max = features.masked_fill(valid <= 0, -torch.inf).amax((2, 3))
    fg_mean, fg_count = weighted(valid * foreground)
    bg_mean, bg_count = weighted(valid * (1-foreground))
    fg_mean = torch.where(fg_count > 0, fg_mean, global_mean)
    bg_mean = torch.where(bg_count > 0, bg_mean, global_mean)
    return torch.cat((global_mean, global_max, fg_mean, bg_mean), 1)


def fit_head(features, labels, weights, protocol):
    """Strictly training-only scaling; fixed regularization and no epoch selection."""
    x = features.double()
    mean = x.mean(0)
    std = x.std(0, unbiased=False).clamp_min(1e-4)
    x = (x-mean)/std
    y = labels.double()
    weights = weights.double()
    weights = weights / weights.sum()
    head = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    optimizer = torch.optim.LBFGS(head.parameters(), lr=1., max_iter=protocol["max_iterations"],
        tolerance_grad=float(protocol["lbfgs_tolerance_grad"]), tolerance_change=1e-10,
        line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(head(x).flatten(), y, reduction="none") * weights).sum()
        loss = loss + float(protocol["l2_half_squared_norm_weight"])*head.weight.square().sum()/2
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        objective = float((F.binary_cross_entropy_with_logits(head(x).flatten(), y, reduction="none")*weights).sum())
    return {"weight":head.weight.detach().flatten(), "bias":head.bias.detach(),
            "mean":mean, "std":std, "training_bce":objective}


def presence_logits(features, head):
    return ((features.double()-head["mean"])/head["std"]) @ head["weight"] + head["bias"]


def positive_threshold(scores, rejection_budget):
    k = math.floor((len(scores)+1)*rejection_budget)
    if not k:
        return -math.inf, k
    return float(scores.sort().values[k-1]), k
