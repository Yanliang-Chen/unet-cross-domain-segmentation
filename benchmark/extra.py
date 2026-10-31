"""Second scoring pass for the feature- and perturbation-based validators.

  python -m benchmark.extra --pool source_only --scenarios source_val clinicdb

For every candidate and every validation image (all images for source_val) it stores:
  z                     pooled U-Net bottleneck features (DEV, IWCV, SND)
  cmr_*_ttd             CMR under test-time dropout on the bottleneck (Talks et al.)
  mixval / mixval_fg    agreement between the prediction on a mixed image and the
                        pseudo-mask of its dominant image (MixVal, Hu et al. 2023)
Outputs <id>.extra.npz next to the main scores; existing files are kept.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F

from benchmark.data import SCENARIOS, build_cache
from benchmark.score import REGISTRY, SCORES, load_models
from benchmark.validators import image_statistics
from utils import choose_device

TTD_P, TTD_SEED, MIX_LAMBDA, BATCH = 0.1, 20261031, 0.6, 64


class Bottleneck:
    """Captures pooled bottleneck features and optionally applies test-time dropout."""

    def __init__(self, models):
        self.dropout, self.features = False, []
        for model in models:
            model.bottleneck.register_forward_hook(self.hook)

    def hook(self, module, inputs, output):
        self.features.append(output.float().mean((2, 3)))
        if self.dropout:
            return F.dropout(output, TTD_P, training=True)
        return output


def forward(models, bottleneck, x, dropout=False):
    bottleneck.dropout, bottleneck.features = dropout, []
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        probs = torch.stack([m(x).float().sigmoid() for m in models]).mean(0)
    features = torch.stack(bottleneck.features).mean(0)
    return torch.logit(probs.clamp(1e-6, 1-1e-6)), features


@torch.inference_mode()
def extra_scenario(models, images, device):
    bottleneck = Bottleneck(models)
    torch.manual_seed(TTD_SEED)
    out = {}
    for start in range(0, len(images), BATCH):
        x = images[start:start+BATCH].to(device).float()/255
        clean, z = forward(models, bottleneck, x)
        ttd, _ = forward(models, bottleneck, x, dropout=True)
        mixed, _ = forward(models, bottleneck, MIX_LAMBDA*x + (1-MIX_LAMBDA)*x.roll(1, 0))
        perturbed = image_statistics(clean, ttd)
        mix = image_statistics(clean, mixed)
        values = {"z": z.half().cpu().numpy(),
                  "cmr_nhd_ttd": perturbed["cmr_nhd"], "cmr_ei_ttd": perturbed["cmr_ei"],
                  "cmr_nhd_fg_ttd": perturbed["cmr_nhd_fg"], "cmr_ei_fg_ttd": perturbed["cmr_ei_fg"],
                  "mixval": mix["cmr_nhd"], "mixval_fg": mix["cmr_nhd_fg"]}
        for key, value in values.items():
            out.setdefault(key, []).append(value)
    return {k: np.concatenate(v) for k, v in out.items()}


def run(pool, scenarios, device):
    candidates = json.loads((REGISTRY/f"{pool}.json").read_text(encoding="utf-8"))
    for scenario in scenarios:
        todo = [c for c in candidates if (c["scenarios"] == "all" or scenario in c["scenarios"])
                and not (SCORES/pool/scenario/f"{c['id']}.extra.npz").exists()]
        if not todo:
            continue
        cache = build_cache(scenario)
        keep = torch.tensor([scenario == "source_val" or r["role"] in ("val", "both") for r in cache["rows"]])
        images = cache["images"][keep].to(device)
        del cache
        (SCORES/pool/scenario).mkdir(parents=True, exist_ok=True)
        began = time.time()
        for k, candidate in enumerate(todo):
            result = extra_scenario(load_models(candidate["checkpoints"], device), images, device)
            tmp = SCORES/pool/scenario/f"{candidate['id']}.extra.tmp.npz"
            np.savez(tmp, **result)
            tmp.replace(SCORES/pool/scenario/f"{candidate['id']}.extra.npz")
            if k % 20 == 0 or k == len(todo)-1:
                print("extra", pool, scenario, f"{k+1}/{len(todo)}", f"{time.time()-began:.0f}s", flush=True)
        del images
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.pool, args.scenarios, choose_device(args.device))


if __name__ == "__main__":
    main()
