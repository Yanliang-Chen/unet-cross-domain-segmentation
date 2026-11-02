"""Qualitative figures: predictions of the deployed models on rule-selected target images.

Usage:
  python -m benchmark.qualitative
Needs the per-image caches of benchmark/postprocess.py. Shown models: base run a0_seed42,
Source-Risk pick, lambda 0.5 (SourceOnly and control lambda 0). Images are chosen by a fixed
rule from those same models, not by eye (ties: first in dataset order):
  gain       polyp image with the largest DACS - SourceOnly Dice difference
  typical    polyp image at the median of that difference
  failure    polyp image with the smallest difference (DACS worse)
  small      small polyp (GT < 2%) at the median DACS Dice among small polyps
  normal ok  (PolypGen, FAR) normal image flagged by the control, clean under DACS and SourceOnly
  normal fp  (PolypGen, FAR) normal image flagged by DACS, clean under SourceOnly
The mean per-image Dice over all deployed picks (6 base runs x lambdas) is listed for context.
Writes results/benchmark/figures/qualitative_<scenario>.png and the chosen ids to
results/benchmark/qualitative.md.
"""
import json
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from torch.nn import functional as F

from benchmark.curves import FIGURES, LABELS, NAMES, TARGETS, load, pick
from benchmark.data import build_cache
from benchmark.postprocess import OUT as CACHE, POOLS
from benchmark.report import RESULTS
from benchmark.score import REGISTRY, SCORES, load_models, predict
from utils import choose_device

SHOWN = ("source_only", "continue", "dacs", "minent_mmd", "fda")
RUN, HEIGHT = "a0_seed42", 256
FILL, CONTOUR = "#2a78d6", "#ffffff"


def per_image(scenario, trajectories, algorithm):
    """Mean over all deployed picks of per-image Dice and any-foreground flag."""
    caches = [np.load(CACHE/scenario/f"{pick(v)['id']}.npz")
              for k, v in trajectories.items() if k[0] == algorithm]
    return np.mean([c["dice"] for c in caches], 0), np.mean([c["any"] for c in caches], 0)


def choose(values, meta, test):
    """values: algorithm -> per-image cache of the shown pick."""
    # polyp-set images with an empty mask (PolypGen) score Dice 0 or 1 only; not shown
    polyp = test & (meta["kind"] == "polyp") & (meta["gt_fraction"] > 0)
    normal = test & (meta["kind"] == "normal")
    diff = values["dacs"]["dice"]-values["source_only"]["dice"]
    order = np.flatnonzero(polyp)[np.argsort(diff[polyp], kind="stable")]
    rows = [("gain", int(order[-1])), ("typical", int(order[len(order)//2])), ("failure", int(order[0]))]
    small = np.flatnonzero(polyp & (meta["gt_fraction"] < 0.02))
    small = small[np.argsort(values["dacs"]["dice"][small], kind="stable")]
    rows.append(("small", int(small[len(small)//2])))
    if normal.any():
        flagged = {a: values[a]["any"].astype(float) for a in ("source_only", "continue", "dacs")}
        ok = flagged["continue"] - flagged["dacs"] - flagged["source_only"]
        fp = flagged["dacs"] - flagged["source_only"]
        rows.append(("normal ok", int(np.flatnonzero(normal)[np.argmax(ok[normal])])))
        rows.append(("normal fp", int(np.flatnonzero(normal)[np.argmax(fp[normal])])))
    return rows


@torch.inference_mode()
def masks_for(checkpoints, images, size, device):
    probs = predict(load_models(checkpoints, device), images.to(device).float()/255).sigmoid()
    return [(F.interpolate(p[None], size=size, mode="bilinear", align_corners=False)[0, 0] > 0.5).cpu().numpy()
            for p in probs]


def draw(ax, image, gt, pred, title):
    ax.imshow(image)
    if pred is not None and pred.any():
        overlay = np.zeros((*pred.shape, 4))
        overlay[pred] = (*matplotlib.colors.to_rgb(FILL), 0.55)
        ax.imshow(overlay)
    if gt is not None and gt.any():
        ax.contour(gt, levels=[0.5], colors=CONTOUR, linewidths=1.2)
    ax.set_title(title, fontsize=8, color="#0b0b0b")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def figure(scenario, registry, device):
    trajectories, _ = load(scenario)
    meta = np.load(SCORES/f"meta_{scenario}.npz")
    cache = build_cache(scenario)
    test = np.array([r["role"] in ("test", "both") for r in cache["rows"]])
    shown = {}
    for a in SHOWN:
        lam = "0.0" if a in ("source_only", "continue") else "0.5"
        row = pick(trajectories[(a, RUN, lam)])
        shown[a] = (row, np.load(CACHE/scenario/f"{row['id']}.npz"))
    rows = choose({a: v for a, (_, v) in shown.items()}, meta, test)
    mean_dice = {a: per_image(scenario, trajectories, a) for a in ("source_only", "dacs")}
    fig, axes = plt.subplots(len(rows), 1+len(SHOWN), figsize=(1.9*(1+len(SHOWN)), 2.0*len(rows)))
    lines = [f"## {NAMES[scenario]}", "", "| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |",
             "|---|---|---:|---|"]
    for r, (label, i) in enumerate(rows):
        h, w = cache["sizes"][i]
        size = (HEIGHT, round(w*HEIGHT/h))
        with Image.open(cache["rows"][i]["image"]) as im:
            image = np.asarray(im.convert("RGB").resize(size[::-1], Image.Resampling.BILINEAR))
        gt = cache["masks"][i]
        gt = None if gt is None else np.asarray(Image.fromarray(gt.numpy()).resize(size[::-1], Image.Resampling.NEAREST))
        kind = "normal" if gt is None else f"GT {100*meta['gt_fraction'][i]:.1f}%"
        draw(axes[r, 0], image, gt, None, f"{label} · {kind}")
        for c, a in enumerate(SHOWN, 1):
            row, values = shown[a]
            pred = masks_for(registry[row["id"]], cache["images"][i:i+1], size, device)[0]
            title = f"{LABELS[a]} {values['dice'][i]:.2f}" if gt is not None else \
                f"{LABELS[a]} {'FP' if values['any'][i] else 'clean'}"
            draw(axes[r, c], image, gt, pred, title)
        name = cache["rows"][i]["image"].replace("\\", "/").split("/")[-1]
        lines.append(f"| {label} | {name} | {100*meta['gt_fraction'][i]:.2f}% | "
                     + ("–" if gt is None else f"{mean_dice['source_only'][0][i]:.2f} / {mean_dice['dacs'][0][i]:.2f}") + " |")
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES/f"qualitative_{scenario}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return lines + [""]


def main():
    device = choose_device("cuda")
    registry = {c["id"]: c["checkpoints"] for pool in POOLS
                for c in json.loads((REGISTRY/f"{pool}.json").read_text(encoding="utf-8"))}
    FIGURES.mkdir(parents=True, exist_ok=True)
    lines = ["# Qualitative examples", "",
             "Rule-selected images (benchmark/qualitative.py docstring), chosen with the shown models: base run "
             f"{RUN}, Source-Risk pick, λ 0.5. Blue = prediction, white contour = ground truth; "
             "titles give that model's native Dice (FP/clean on normal images).", ""]
    for scenario in TARGETS:
        lines += figure(scenario, registry, device)
        print("wrote", scenario, flush=True)
    (RESULTS/"qualitative.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
