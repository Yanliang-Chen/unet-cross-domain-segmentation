"""Label-free validators for binary segmentation; every score is oriented higher = better.

Per-image statistics are computed once per checkpoint and pooled afterwards, so a
validator never needs target masks. Pixel-level versions of the classification
validators in Xiong et al. pool all target pixels; CMR follows Talks et al.
"""
import math
import numpy as np
import torch

EPS = 1e-6
MCC_TEMPERATURE = 2.5
CMR_KEYS = ("cmr_nhd", "cmr_ei", "cmr_nhd_fg", "cmr_ei_fg")
STAT_KEYS = ("entropy", "mean_p", "fg", "g00", "g01", "g11", "m00", "m01", "m11") + CMR_KEYS


def binary_entropy(p):
    p = p.clamp(EPS, 1-EPS)
    return -(p*p.log() + (1-p)*(1-p).log())


def _restricted(clean_mask, noisy_mask, value):
    """Mean of value over pixels predicted as the class in either map; NaN if none."""
    union = clean_mask | noisy_mask
    count = union.sum(1).float()
    total = (value*union).sum(1)
    return torch.where(count > 0, total/count.clamp_min(1), torch.full_like(total, float("nan")))


@torch.no_grad()
def image_statistics(logits, noisy_logits, threshold=0.5):
    """logits, noisy_logits: (B,1,H,W) on clean and Gaussian-perturbed input."""
    p = logits.float().sigmoid().flatten(1)
    q = noisy_logits.float().sigmoid().flatten(1)
    y = (logits.float()/MCC_TEMPERATURE).sigmoid().flatten(1)
    w = 1 + torch.exp(-binary_entropy(y))
    a, b = p > threshold, q > threshold
    agree = (a == b).float()
    fg_iou = _restricted(a, b, (a & b).float())
    bg_iou = _restricted(~a, ~b, (~a & ~b).float())
    fg_ei = _restricted(a, b, (p*q).sqrt()*agree)
    bg_ei = _restricted(~a, ~b, ((1-p)*(1-q)).sqrt()*agree)
    stats = {
        "entropy": binary_entropy(p).mean(1), "mean_p": p.mean(1), "fg": a.float().mean(1),
        "g00": ((1-p)**2).mean(1), "g01": ((1-p)*p).mean(1), "g11": (p**2).mean(1),
        "m00": (w*(1-y)**2).mean(1), "m01": (w*(1-y)*y).mean(1), "m11": (w*y**2).mean(1),
        "cmr_nhd": torch.stack((fg_iou, bg_iou), 1).nanmean(1),
        "cmr_ei": torch.stack((fg_ei, bg_ei), 1).nanmean(1),
        "cmr_nhd_fg": fg_iou.nan_to_num(0.0),       # foreground only; empty/empty scores 0
        "cmr_ei_fg": fg_ei.nan_to_num(0.0),
    }
    return {k: v.cpu().numpy() for k, v in stats.items()}


def _entropy_scalar(p):
    p = min(max(p, EPS), 1-EPS)
    return -(p*math.log(p) + (1-p)*math.log(1-p))


def validator_scores(stats):
    """stats: dict of per-image arrays over the validation images of one scenario."""
    mean = {k: float(np.mean(stats[k])) for k in STAT_KEYS if k not in CMR_KEYS}
    gram = np.array([[mean["g00"], mean["g01"]], [mean["g01"], mean["g11"]]])
    confusion = np.array([[mean["m00"], mean["m01"]], [mean["m01"], mean["m11"]]])
    confusion = confusion/confusion.sum(1, keepdims=True)
    return {
        "entropy": -mean["entropy"],
        "infomax": _entropy_scalar(mean["mean_p"]) - mean["entropy"],
        "bnm": float(np.sqrt(np.clip(np.linalg.eigvalsh(gram), 0, None)).sum()),
        "mcc": -float(confusion[0, 1] + confusion[1, 0])/2,
        "cmr_nhd_gauss": float(np.median(stats["cmr_nhd"])),
        "cmr_ei_gauss": float(np.median(stats["cmr_ei"])),
        "cmr_nhd_fg_gauss": float(np.median(stats["cmr_nhd_fg"])),
        "cmr_ei_fg_gauss": float(np.median(stats["cmr_ei_fg"])),
        "foreground_fraction": mean["fg"],
    }


def auroc(positive_scores, negative_scores):
    """Mann-Whitney AUROC with average ranks for ties."""
    from scipy.stats import rankdata
    pos, neg = np.asarray(positive_scores), np.asarray(negative_scores)
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = rankdata(np.concatenate((pos, neg)))
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


FEATURE_SAMPLE, FEATURE_SEED, SND_TEMPERATURE, FOLDS = 1000, 20261031, 0.05, 5


def _subsample(z, rng):
    return z if len(z) <= FEATURE_SAMPLE else z[rng.choice(len(z), FEATURE_SAMPLE, replace=False)]


def _logistic_probability(train_x, train_y, test_x, l2=1e-2):
    from scipy.optimize import minimize
    def loss(w):
        logits = train_x @ w[:-1] + w[-1]
        value = np.logaddexp(0, logits).mean() - (train_y*logits).mean() + l2*(w[:-1]**2).sum()/2
        p = 1/(1+np.exp(-logits))
        grad = np.append(train_x.T @ (p-train_y)/len(train_y) + l2*w[:-1], (p-train_y).mean())
        return value, grad
    w = minimize(loss, np.zeros(train_x.shape[1]+1), jac=True, method="L-BFGS-B").x
    return 1/(1+np.exp(-(test_x @ w[:-1] + w[-1])))


def importance_weights(z_source, z_target, rng):
    """Out-of-fold density ratio p(target|z)/p(source|z) for source validation images."""
    z_target = _subsample(z_target, rng)
    both = np.concatenate((z_source, z_target)).astype(np.float64)
    both = (both - both.mean(0))/(both.std(0) + 1e-6)
    xs, xt = both[:len(z_source)], both[len(z_source):]
    folds = rng.permutation(len(xs)) % FOLDS
    p = np.empty(len(xs))
    for fold in range(FOLDS):
        held = folds == fold
        train_x = np.concatenate((xs[~held], xt))
        train_y = np.concatenate((np.zeros((~held).sum()), np.ones(len(xt))))
        p[held] = _logistic_probability(train_x, train_y, xs[held])
    p = p.clip(1e-4, 1-1e-4)
    return p/(1-p)*len(xs)/len(xt)


def _dev_risk(weights, loss):
    weighted = weights*loss
    variance = np.var(weights)
    eta = -np.cov(weighted, weights)[0, 1]/variance if variance > 0 else 0.0
    return float(weighted.mean() + eta*weights.mean() - eta)


def feature_validators(z_source, source_dice, z_target, mix, ttd):
    """DEV / DEV-N / IWCV (You et al. 2019; loss = 1 - native source Dice), SND (Saito et al.
    2021) on pooled bottleneck features, MixVal and CMR-TTD medians; higher = better."""
    rng = np.random.default_rng(FEATURE_SEED)
    weights = importance_weights(z_source.astype(np.float64), z_target.astype(np.float64), rng)
    loss = 1 - np.asarray(source_dice, float)
    zt = _subsample(z_target.astype(np.float64), rng)
    zt = zt/np.linalg.norm(zt, axis=1, keepdims=True).clip(1e-6)
    similarity = zt @ zt.T/SND_TEMPERATURE
    np.fill_diagonal(similarity, -np.inf)
    similarity -= similarity.max(1, keepdims=True)
    prob = np.exp(similarity)
    prob /= prob.sum(1, keepdims=True)
    snd = float(-(prob*np.log(prob.clip(1e-12))).sum(1).mean())
    return {"iwcv": -float((weights*loss).mean()), "dev": -_dev_risk(weights, loss),
            "devn": -_dev_risk(weights/weights.mean(), loss), "snd": snd,
            "mixval": float(np.median(mix["mixval"])), "mixval_fg": float(np.median(mix["mixval_fg"])),
            **{f"{k}_ttd": float(np.median(ttd[f"{k}_ttd"])) for k in ("cmr_nhd", "cmr_ei", "cmr_nhd_fg", "cmr_ei_fg")}}
