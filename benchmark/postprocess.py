"""Small-component removal on the deployed models (benchmark_protocol.yaml: postprocessing).

Usage:
  python -m benchmark.postprocess --scenarios clinicdb etis colondb polypgen far
Each trajectory's Source-Risk pick is predicted once at native resolution; predicted
components smaller than min_area x image area are removed. Per-image results are cached in
runs/benchmark/postprocess/<scenario>/<id>.npz; the table goes to
results/benchmark/postprocessing.md.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F

from benchmark.analyze import SMALL, load_pools
from benchmark.data import build_cache
from benchmark.report import BOOTSTRAP, RESULTS, SEED, trajectories
from benchmark.score import BATCH, REGISTRY, SCORES, load_models, predict
from utils import choose_device

OUT = Path("runs/benchmark/postprocess")
POOLS = ("source_only", "uda_continue", "uda_dacs", "uda_minent_mmd", "uda_fda")
ORDER = ("source_only", "continue", "dacs", "minent_mmd", "fda")
MIN_AREAS = (0.001, 0.0041)
EIGHT = np.ones((3, 3), bool)


def remove_small(mask, min_area):
    if not mask.any():
        return mask
    labels, n = ndimage.label(mask, structure=EIGHT)
    if n == 0:
        return mask
    keep = np.bincount(labels.ravel()) >= min_area*mask.size
    keep[0] = False
    return keep[labels]


def dice(pred, gt):
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2*np.logical_and(pred, gt).sum()/total


@torch.inference_mode()
def predict_variants(checkpoints, cache, device):
    """Dice (polyp images) and any-foreground flag per image, raw and after each min_area."""
    models = load_models(checkpoints, device)
    n = len(cache["rows"])
    suffixes = ["", *(f"_{a:g}" for a in MIN_AREAS)]
    out = {f"dice{s}": np.full(n, np.nan) for s in suffixes}
    out.update({f"any{s}": np.zeros(n, bool) for s in suffixes})
    for start in range(0, n, BATCH):
        probs = predict(models, cache["gpu_images"][start:start+BATCH].float()/255).sigmoid()
        for j in range(len(probs)):
            i = start+j
            mask = (F.interpolate(probs[j:j+1], size=cache["sizes"][i], mode="bilinear",
                                  align_corners=False)[0, 0] > 0.5).cpu().numpy()
            variants = [mask, *(remove_small(mask, a) for a in MIN_AREAS)]
            for s, m in zip(suffixes, variants):
                out[f"any{s}"][i] = m.any()
                if cache["masks"][i] is not None:
                    out[f"dice{s}"][i] = dice(m, cache["masks"][i])
    return out


def delta(diffs, keys, groups, rng):
    """Mean per-group difference (lambdas averaged per base run) and joint bootstrap interval."""
    if groups.size and all(groups):
        labels, inverse = np.unique(groups, return_inverse=True)
    else:
        labels, inverse = np.arange(len(groups)), np.arange(len(groups))
    runs = sorted({k[1] for k in keys})
    per_run = np.stack([np.mean([np.bincount(inverse, diffs[k], len(labels))
                                 / np.bincount(inverse, minlength=len(labels))
                                 for k in keys if k[1] == b], 0) for b in runs])
    draws = per_run[rng.integers(0, len(runs), (BOOTSTRAP, len(runs)))]
    pick = rng.integers(0, len(labels), (BOOTSTRAP, len(labels)))
    means = np.take_along_axis(draws, pick[:, None, :], 2).mean((1, 2))
    return float(per_run.mean()), np.quantile(means, [0.025, 0.975])


def fmt(value, ci=None, scale=1.0, digits=3):
    text = f"{value*scale:+.{digits}f}"
    return text if ci is None else f"{text} [{ci[0]*scale:+.{digits}f}, {ci[1]*scale:+.{digits}f}]"


def scenario_table(scenario, rows, registry, device, rng):
    cache = build_cache(scenario)
    meta = np.load(SCORES/f"meta_{scenario}.npz")
    test = np.array([r["role"] in ("test", "both") for r in cache["rows"]])
    polyp, normal = test & (meta["kind"] == "polyp"), test & (meta["kind"] == "normal")
    small = polyp & (meta["gt_fraction"] < SMALL)
    picks = {k: max(v, key=lambda r: r["validators"]["source_risk"]) for k, v in trajectories(rows).items()}
    cache["gpu_images"] = cache["images"].to(device)
    cache["masks"] = [m.numpy() if m is not None else None for m in cache["masks"]]
    (OUT/scenario).mkdir(parents=True, exist_ok=True)
    results = {}
    for n, (key, row) in enumerate(sorted(picks.items())):
        path = OUT/scenario/f"{row['id']}.npz"
        if not path.exists():
            np.savez(path, **predict_variants(registry[row["id"]], cache, device))
        results[key] = dict(np.load(path))
        drift = np.abs(results[key]["dice"][polyp] - row["_dice"]).max()
        print(scenario, f"{n+1}/{len(picks)}", row["id"], f"max Dice drift vs score {drift:.4f}", flush=True)
    del cache
    torch.cuda.empty_cache()

    head = ["| Algorithm | Picks | Dice | ΔDice 0.1% [95% CI] | ΔDice 0.41% | Small Dice → 0.1% |"]
    if normal.any():
        head[0] += " Normal FPR | ΔFPR 0.1% [95% CI] (points) | ΔFPR 0.41% |"
    head.append("|" + "---|"*(head[0].count("|")-1))
    lines = [f"## {scenario}", "", *head]
    for algorithm in ORDER:
        keys = [k for k in results if k[0] == algorithm]
        if not keys:
            continue
        mean = lambda name, where: float(np.mean([results[k][name][where].mean() for k in keys]))
        d1 = delta({k: results[k]["dice_0.001"][polyp]-results[k]["dice"][polyp] for k in keys},
                   keys, meta["group"][polyp], rng)
        d2 = delta({k: results[k]["dice_0.0041"][polyp]-results[k]["dice"][polyp] for k in keys},
                   keys, meta["group"][polyp], rng)
        line = (f"| {algorithm} | {len(keys)} | {mean('dice', polyp):.3f} | {fmt(*d1)} | {fmt(d2[0])} | "
                f"{mean('dice', small):.3f} → {mean('dice_0.001', small):.3f} |")
        if normal.any():
            f1 = delta({k: results[k]["any_0.001"][normal].astype(float)-results[k]["any"][normal] for k in keys},
                       keys, meta["group"][normal], rng)
            f2 = delta({k: results[k]["any_0.0041"][normal].astype(float)-results[k]["any"][normal] for k in keys},
                       keys, meta["group"][normal], rng)
            line += f" {100*mean('any', normal):.1f}% | {fmt(*f1, scale=100, digits=1)} | {fmt(f2[0], scale=100, digits=1)} |"
        lines.append(line)
    return lines + [""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["clinicdb", "etis", "colondb", "polypgen", "far"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = choose_device(args.device)
    table, _ = load_pools(POOLS, features=False)
    registry = {c["id"]: c["checkpoints"] for pool in POOLS
                for c in json.loads((REGISTRY/f"{pool}.json").read_text(encoding="utf-8"))}
    rng = np.random.default_rng(SEED)
    lines = ["# Small-component removal on deployed models", "",
             "Each trajectory's Source-Risk pick; predicted 8-connected components smaller than 0.1% "
             "(primary) or 0.41% (sensitivity) of the native image are removed. Δ = after − before, "
             "per-group mean with lambdas averaged per base run; 95% interval from "
             f"{BOOTSTRAP} joint bootstrap draws over base runs and image groups. "
             "Small Dice = polyps below 2% of the image. Normal FPR = share of normal images with any "
             "predicted pixel.", ""]
    for scenario in args.scenarios:
        lines += scenario_table(scenario, table[scenario], registry, device, rng)
    (RESULTS/"postprocessing.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RESULTS/"postprocessing.md")


if __name__ == "__main__":
    main()
