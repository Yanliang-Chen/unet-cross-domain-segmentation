"""Score candidate checkpoints: native-resolution test metrics and validator statistics.

Usage:
  python -m benchmark.score register-source        # 180 existing source-only checkpoints
  python -m benchmark.score run --pool source_only [--scenarios clinicdb etis]
Outputs one npz per candidate in runs/benchmark/scores/<pool>/; existing files are kept.
"""
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from benchmark.data import SCENARIOS, build_cache
from benchmark.validators import image_statistics
from models.unet import build_model
from scripts.check_protocol import sha256
from utils import choose_device

REGISTRY = Path("runs/benchmark/candidates")
SCORES = Path("runs/benchmark/scores")
SIGMA, NOISE_SEED, BATCH = 0.05, 20261031, 64


def register_source():
    rows = []
    for run in sorted(Path("runs/real").glob("a[01]_seed4[234]")):
        augmentation, seed = re.fullmatch(r"(a[01])_seed(\d+)", run.name).groups()
        for path in sorted((run/"checkpoints").glob("epoch_*.pt")):
            epoch = int(path.stem.split("_")[1])
            rows.append({"id": f"source_{run.name}_e{epoch:03d}", "algorithm": "source_only",
                         "lam": 0.0, "augmentation": augmentation, "seed": int(seed),
                         "step": epoch, "run": run.name, "checkpoints": [str(path)],
                         "scenarios": "all"})
    if len(rows) != 180:
        raise ValueError(f"Expected 180 source checkpoints, found {len(rows)}")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    (REGISTRY/"source_only.json").write_text(json.dumps(rows, indent=1)+"\n", encoding="utf-8")
    print("registered", len(rows))


def load_models(paths, device):
    models = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = build_model(state["config"])
        model.load_state_dict(state["model_state_dict"])
        models.append(model.to(device).eval())
    return models


def predict(models, x):
    """Mean probability of the models, returned as logits (single model: unchanged)."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
        logits = [m(x).float() for m in models]
    if len(logits) == 1:
        return logits[0]
    return torch.logit(torch.stack([l.sigmoid() for l in logits]).mean(0).clamp(1e-6, 1-1e-6))


def _native_metrics(logits, masks, sizes):
    """Per-image native metrics; masks entries are GPU bool tensors or None (normal/unknown)."""
    out = {k: np.full(len(sizes), np.nan) for k in
           ("dice", "iou", "precision", "recall", "fg_native", "max_prob")}
    probs = logits.sigmoid()
    out["max_prob"] = probs.flatten(1).amax(1).cpu().numpy()
    by_size = {}
    for i, size in enumerate(sizes):
        by_size.setdefault(size, []).append(i)
    for size, index in by_size.items():
        up = F.interpolate(probs[index], size=size, mode="bilinear", align_corners=False)[:, 0] > 0.5
        predicted = up.flatten(1).sum(1).float()
        out["fg_native"][index] = (predicted/up[0].numel()).cpu().numpy()
        for j, i in enumerate(index):
            if masks[i] is None:
                continue
            gt = masks[i]
            tp = (up[j] & gt).sum().float()
            pred, actual = predicted[j], gt.sum().float()
            values = torch.stack((
                torch.where(pred+actual == 0, 1.0, 2*tp/(pred+actual).clamp_min(1)),
                torch.where(pred+actual-tp == 0, 1.0, tp/(pred+actual-tp).clamp_min(1)),
                torch.where(pred == 0, (actual == 0).float(), tp/pred.clamp_min(1)),
                torch.where(actual == 0, 1.0, tp/actual.clamp_min(1)))).cpu().numpy()
            for key, value in zip(("dice", "iou", "precision", "recall"), values):
                out[key][i] = value
    return out


@torch.inference_mode()
def score_scenario(models, cache, device):
    rows, n = cache["rows"], len(cache["rows"])
    images = cache["gpu_images"]
    validation = np.array([r["role"] in ("val", "both") for r in rows])
    test = np.array([r["role"] in ("test", "both") for r in rows])
    generator = torch.Generator(device=device).manual_seed(NOISE_SEED)
    result = {}
    for start in range(0, n, BATCH):
        stop = min(start+BATCH, n)
        x = images[start:stop].float()/255
        logits = predict(models, x)
        noise = torch.randn(x.shape, generator=generator, device=device)*SIGMA
        noisy = predict(models, (x+noise).clamp(0, 1))
        stats = image_statistics(logits, noisy)
        native = _native_metrics(logits, cache["gpu_masks"][start:stop], cache["sizes"][start:stop])
        for key, value in {**stats, **native}.items():
            result.setdefault(key, []).append(value)
    result = {k: np.concatenate(v) for k, v in result.items()}
    result["is_validation"], result["is_test"] = validation, test
    return result


def scenario_meta(cache):
    rows = cache["rows"]
    gt = [float(m.float().mean()) if m is not None else (0.0 if r["kind"] == "normal" else np.nan)
          for r, m in zip(rows, cache["masks"])]
    return {"kind": np.array([r["kind"] for r in rows]), "group": np.array([r["group"] for r in rows]),
            "gt_fraction": np.array(gt)}


def run(pool, scenarios, device):
    candidates = json.loads((REGISTRY/f"{pool}.json").read_text(encoding="utf-8"))
    out_dir = SCORES/pool
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        todo = [c for c in candidates if (c["scenarios"] == "all" or scenario in c["scenarios"])
                and not (out_dir/scenario/f"{c['id']}.npz").exists()]
        if not todo:
            continue
        cache = build_cache(scenario)
        meta_path = SCORES/f"meta_{scenario}.npz"
        if not meta_path.exists():
            np.savez(meta_path, **scenario_meta(cache))
        cache["gpu_images"] = cache["images"].to(device)
        cache["gpu_masks"] = [m.to(device) if m is not None else None for m in cache["masks"]]
        (out_dir/scenario).mkdir(exist_ok=True)
        began = time.time()
        for k, candidate in enumerate(todo):
            models = load_models(candidate["checkpoints"], device)
            result = score_scenario(models, cache, device)
            tmp = out_dir/scenario/f"{candidate['id']}.tmp.npz"
            np.savez(tmp, **result)
            tmp.replace(out_dir/scenario/f"{candidate['id']}.npz")
            if k % 10 == 0 or k == len(todo)-1:
                print(scenario, f"{k+1}/{len(todo)}", f"{time.time()-began:.0f}s", flush=True)
        del cache
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register-source")
    r = sub.add_parser("run")
    r.add_argument("--pool", required=True)
    r.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    r.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "register-source":
        register_source()
    else:
        print("protocol", sha256("benchmark_protocol.yaml")[:12], flush=True)
        run(args.pool, args.scenarios, choose_device(args.device))


if __name__ == "__main__":
    main()
