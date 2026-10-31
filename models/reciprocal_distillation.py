"""Binary transfer of ReDi-FM equations 6--16 to the source-anchored pipeline."""
import math
import torch
from torch.nn import functional as F


def confidence(probability, protocol):
    variance = float(protocol["confidence_variance"])
    return float(protocol["confidence_gamma"]) - torch.exp(
        -(probability-float(protocol["confidence_mu"]))**2/(2*variance)) / math.sqrt(2*math.pi*variance)


def weighted_hard(logits, probability, protocol):
    weight = confidence(probability, protocol)
    label = (probability > .5).float()
    pred = logits.float().sigmoid()
    bce = (F.binary_cross_entropy_with_logits(logits.float(), label, reduction="none")*weight).mean((1,2,3))
    intersection = (weight*pred*label).sum((1,2,3))
    denominator = (weight*pred).sum((1,2,3))+(weight*label).sum((1,2,3))
    return bce+1-(2*intersection+1)/(denominator+1)


def reciprocal_loss(logits, source, foundation, valid, protocol):
    source, foundation = source.detach().float(), foundation.detach().float()
    pred = logits.float().sigmoid()
    ys, yf, yt = source > .5, foundation > .5, pred.detach() > .5
    agreement = (2*(ys & yf).sum((1,2,3)).float()+1e-6) / (
        ys.sum((1,2,3))+yf.sum((1,2,3))+1e-6)
    hard_source = weighted_hard(logits, source, protocol)
    hard_foundation = weighted_hard(logits, foundation, protocol)
    src_two = torch.cat((1-source, source), 1).clamp(1e-6, 1-1e-6)
    fm_two = torch.cat((1-foundation, foundation), 1).clamp(1e-6, 1-1e-6)
    pred_two = torch.cat((1-pred, pred), 1)
    source_classes = torch.cat((~ys, ys), 1)
    foundation_classes = torch.cat((~yf, yf), 1)
    target_classes = torch.cat((~yt, yt), 1)
    normalized_entropy = -(src_two*src_two.log()).sum(1, keepdim=True)/math.log(2)
    hard_region = source_classes | (normalized_entropy >= float(protocol["uncertain_normalized_entropy_threshold"]))
    eta = (foundation_classes.float().mean((2,3), keepdim=True)
           +(target_classes != foundation_classes).float().mean((2,3), keepdim=True))/2
    alpha = torch.where(hard_region, eta, 1-eta)
    r = agreement[:,None,None,None]
    alpha = r*alpha+(1-r)
    # Equation 12 uses p_target * log(p_foundation), not standard forward CE.
    soft = -(alpha*pred_two*fm_two.log()).sum(1).mean((1,2))
    per_image = hard_foundation+agreement*hard_source+soft
    valid = valid.float()
    loss = (per_image*valid).sum()/valid.sum().clamp_min(1)
    return loss, {"agreement": float((agreement*valid).sum()/valid.sum().clamp_min(1)),
                  "valid_fraction": float(valid.mean())}
