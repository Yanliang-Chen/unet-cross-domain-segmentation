"""Lambda sensitivity, training-step curves and small-polyp Dice from the tracked candidate tables.

Usage:
  python -m benchmark.curves
Reads results/benchmark/<scenario>/candidates.csv only (no GPU, no runs/), writes
results/benchmark/curves.md and results/benchmark/figures/{step_curves,step_fpr,lambda}.png.
Deployment is the same as in report.py: the Source-Risk pick of each trajectory.
"""
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from benchmark.report import RESULTS

TARGETS = ("clinicdb", "etis", "colondb", "polypgen", "far")
NAMES = {"clinicdb": "ClinicDB", "etis": "ETIS", "colondb": "ColonDB", "polypgen": "PolypGen", "far": "FAR"}
MAIN = ("dacs", "minent_mmd", "fda")
LABELS = {"dacs": "DACS", "minent_mmd": "MinEnt+MMD", "fda": "FDA", "continue": "Control",
          "source_only": "SourceOnly", "adabn": "AdaBN", "minent": "MinEnt", "mmd": "MMD", "dann": "DANN",
          "advent": "AdvEnt", "bnm": "BNM", "mcc": "MCC", "meanteacher": "Mean Teacher"}
LAMS = ("0.1", "0.5", "1.0")
COLORS = {"dacs": "#2a78d6", "minent_mmd": "#eb6834", "fda": "#1baf7a", "continue": "#8a8984"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIGURES = RESULTS/"figures"


def number(value):
    return float(value) if value not in ("", None) else np.nan


def load(scenario):
    rows = list(csv.DictReader((RESULTS/scenario/"candidates.csv").open(encoding="utf-8")))
    for r in rows:
        r["step"] = int(r["step"])
        for key in ("dice", "small_dice", "fpr_any", "source_risk"):
            r[key] = number(r[key])
    trajectories = defaultdict(list)
    for r in rows:
        trajectories[(r["algorithm"], r["base_run"], r["lam"])].append(r)
    for v in trajectories.values():
        v.sort(key=lambda r: r["step"])
    # warm start of every adaptation trajectory = SourceOnly epoch 30 of the same base run
    start = {b: v[-1] for (a, b, _), v in trajectories.items() if a == "source_only"}
    return trajectories, start


def pick(rows):
    return max(rows, key=lambda r: r["source_risk"])


def mean(values):
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else np.nan


def fmt(value, digits=3, percent=False):
    if np.isnan(value):
        return "–"
    return f"{100*value:.1f}%" if percent else f"{value:.{digits}f}"


def lambda_table(data):
    lines = ["## 1. Lambda sensitivity (deployed Source-Risk pick, mean over 6 base runs)", "",
             "Cells: Dice at λ 0.1 / 0.5 / 1; PolypGen also normal-image FPR. FAR has λ=0.5 only.", "",
             "| Target | Method | Dice λ 0.1 | λ 0.5 | λ 1 | Normal FPR λ 0.1 | λ 0.5 | λ 1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in TARGETS[:4]:
        trajectories, _ = data[s]
        for a in MAIN:
            cells = [[pick(v) for (al, _, l), v in trajectories.items() if al == a and l == lam] for lam in LAMS]
            dice = [fmt(mean([p["dice"] for p in c])) for c in cells]
            fpr = [fmt(mean([p["fpr_any"] for p in c]), percent=True) for c in cells]
            lines.append(f"| {NAMES[s]} | {LABELS[a]} | {' | '.join(dice)} | {' | '.join(fpr)} |")
    return lines + [""]


def step_table(data):
    lines = ["## 2. Training length (all λ; FAR λ=0.5)", "",
             "Per trajectory: start = warm start (SourceOnly epoch 30); last = step 2000; pick = Source-Risk "
             "pick; best = best saved checkpoint by target Dice (needs labels); regret = best − pick. "
             "Means over trajectories.", "",
             "| Target | Method | Start | Last | Pick | Best | Regret | Mean pick step | Pick = last |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in TARGETS:
        trajectories, start = data[s]
        for a in (*MAIN, "continue"):
            keys = [k for k in trajectories if k[0] == a]
            picks = [pick(trajectories[k]) for k in keys]
            best = [max(r["dice"] for r in trajectories[k]) for k in keys]
            last = [trajectories[k][-1] for k in keys]
            lines.append(f"| {NAMES[s]} | {LABELS[a]} | {fmt(mean([start[k[1]]['dice'] for k in keys]))} | "
                         f"{fmt(mean([r['dice'] for r in last]))} | {fmt(mean([p['dice'] for p in picks]))} | "
                         f"{fmt(mean(best))} | {fmt(mean(best)-mean([p['dice'] for p in picks]))} | "
                         f"{np.mean([p['step'] for p in picks]):.0f} | "
                         f"{sum(p is l for p, l in zip(picks, last))}/{len(keys)} |")
    return lines + [""]


def small_table(data, counts):
    algorithms = ("source_only", "continue", "adabn", "dacs", "minent_mmd", "fda", "minent", "mmd", "dann")
    lines = ["## 3. Small-polyp Dice (GT below 2% of the image; deployed picks, all λ)", "",
             "Cells: small-polyp Dice (all-polyp Dice). Gain columns: mean over trajectories of the "
             "difference to the same base run's SourceOnly pick, with wins.", "",
             "| Method | " + " | ".join(f"{NAMES[s]} (n={counts.get(s, '?')})" for s in TARGETS) + " |",
             "|---|" + "---:|"*len(TARGETS)]
    gains = {}
    for a in algorithms:
        cells = []
        for s in TARGETS:
            trajectories, _ = data[s]
            source = {k[1]: pick(v) for k, v in trajectories.items() if k[0] == "source_only"}
            picks = [(k[1], pick(v)) for k, v in trajectories.items() if k[0] == a]
            if not picks:
                cells.append("–")
                continue
            cells.append(f"{fmt(mean([p['small_dice'] for _, p in picks]))} "
                         f"({fmt(mean([p['dice'] for _, p in picks]))})")
            if a in (*MAIN, "continue"):
                small = [p["small_dice"]-source[b]["small_dice"] for b, p in picks]
                whole = [p["dice"]-source[b]["dice"] for b, p in picks]
                gains[(a, s)] = (np.mean(small), sum(d > 0 for d in small), len(small), np.mean(whole))
        lines.append(f"| {LABELS[a]} | " + " | ".join(cells) + " |")
    lines += ["", "| Method | " + " | ".join(NAMES[s] for s in TARGETS) + " |", "|---|" + "---|"*len(TARGETS)]
    for a in (*MAIN, "continue"):
        cells = [f"{g[0]:+.3f} ({g[1]}/{g[2]}; all {g[3]:+.3f})" for g in (gains[(a, s)] for s in TARGETS)]
        lines.append(f"| {LABELS[a]} gain vs SourceOnly | " + " | ".join(cells) + " |")
    return lines + [""]


def style(ax, title):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def curve(trajectories, start, algorithm, metric):
    """Mean over base runs of metric by step at λ=0.5 (control λ=0), step 0 = warm start."""
    lam = "0.0" if algorithm == "continue" else "0.5"
    keys = [k for k in trajectories if k[0] == algorithm and k[2] == lam]
    steps = [0, *(r["step"] for r in trajectories[keys[0]])]
    values = np.array([[start[k[1]][metric], *(r[metric] for r in trajectories[k])] for k in keys])
    return steps, values.mean(0)


def step_figure(data, targets, metric, path, ylabel):
    fig, axes = plt.subplots(1, len(targets), figsize=(2.9*len(targets), 2.8), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), targets):
        trajectories, start = data[s]
        scale = 100 if metric == "fpr_any" else 1
        source = mean([pick(v)[metric] for k, v in trajectories.items() if k[0] == "source_only"])
        ax.axhline(scale*source, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for a in (*MAIN, "continue"):
            steps, values = curve(trajectories, start, a, metric)
            ax.plot(steps, scale*values, color=COLORS[a], linewidth=2, marker="o", markersize=3.5,
                    label=LABELS[a], linestyle="-" if a != "continue" else (0, (1, 1.5)))
        style(ax, NAMES[s])
        ax.set_xticks([0, 1000, 2000])
        ax.set_xlabel("adaptation step", color=MUTED, fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel(ylabel, color=MUTED, fontsize=8)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle=(0, (4, 3))))
    labels.append("SourceOnly (deployed)")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def lambda_figure(data, path):
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 2.8))
    panels = [(s, "dice") for s in TARGETS[:4]] + [("polypgen", "fpr_any")]
    for ax, (s, metric) in zip(axes, panels):
        trajectories, _ = data[s]
        scale = 100 if metric == "fpr_any" else 1
        source = mean([pick(v)[metric] for k, v in trajectories.items() if k[0] == "source_only"])
        ax.axhline(scale*source, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label="SourceOnly (deployed)")
        for a in MAIN:
            y = [scale*mean([pick(v)[metric] for (al, _, l), v in trajectories.items() if al == a and l == lam])
                 for lam in LAMS]
            ax.plot([0.1, 0.5, 1.0], y, color=COLORS[a], linewidth=2, marker="o", markersize=4, label=LABELS[a])
        style(ax, f"{NAMES[s]} — {'normal FPR (%)' if metric == 'fpr_any' else 'deployed Dice'}")
        ax.set_xticks([0.1, 0.5, 1.0])
        ax.set_xlabel("λ", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[1:]+handles[:1], labels[1:]+labels[:1], loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def small_counts():
    try:
        from benchmark.score import SCORES
        out = {}
        for s in TARGETS:
            meta = np.load(SCORES/f"meta_{s}.npz")
            out[s] = int(((meta["kind"] == "polyp") & (meta["gt_fraction"] < 0.02)).sum())
        return out
    except (OSError, ImportError):
        return {}


def main():
    data = {s: load(s) for s in TARGETS}
    FIGURES.mkdir(parents=True, exist_ok=True)
    lines = ["# Lambda, training length and small polyps", "",
             "From results/benchmark/<target>/candidates.csv (benchmark/curves.py). Figures: "
             "`figures/step_curves.png`, `figures/step_fpr.png`, `figures/lambda.png`.", ""]
    lines += lambda_table(data) + step_table(data) + small_table(data, small_counts())
    (RESULTS/"curves.md").write_text("\n".join(lines), encoding="utf-8")
    step_figure(data, TARGETS, "dice", FIGURES/"step_curves.png", "target Dice (mean of 6 runs, λ=0.5)")
    step_figure(data, ("polypgen", "far"), "fpr_any", FIGURES/"step_fpr.png", "normal-image FPR (%)")
    lambda_figure(data, FIGURES/"lambda.png")
    print("wrote", RESULTS/"curves.md", FIGURES)


if __name__ == "__main__":
    main()
