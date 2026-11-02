"""Summary figures for the report, from tables already in results/benchmark/.

Usage:
  python -m benchmark.summary_figures
gains.png      paired gain (95% interval) of each method over SourceOnly and over the control,
               per target, read from results/benchmark/<target>/paired.md
dice_fpr.png   deployed Dice against normal-image FPR on PolypGen and FAR: one small dot per
               trajectory (Source-Risk pick), one large marker per method mean, from candidates.csv
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from benchmark.curves import COLORS, FIGURES, GRID, INK, LABELS, MAIN, MUTED, NAMES, TARGETS, load, pick, style
from benchmark.report import RESULTS

INTERVAL = re.compile(r"([+-]\d\.\d+) \[([+-]\d\.\d+), ([+-]\d\.\d+)\]")
SOURCE = "#3d3c39"


def paired(scenario):
    """algorithm -> {"control": (gain, low, high), "source": (...)} from paired.md."""
    out = {}
    for line in (RESULTS/scenario/"paired.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        found = INTERVAL.findall(line)
        if len(found) == 2:
            out[cells[0]] = {name: tuple(map(float, f)) for name, f in zip(("control", "source"), found)}
    return out


def gains_figure(path):
    tables = {s: paired(s) for s in TARGETS}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    panels = (("source", "Gain over SourceOnly", (*MAIN, "continue")), ("control", "Gain over continued-training control", MAIN))
    for ax, (ref, title, algorithms) in zip(axes, panels):
        ax.axhline(0, color=MUTED, linewidth=1)
        width = 0.75/len(algorithms)
        for j, a in enumerate(algorithms):
            x = np.arange(len(TARGETS)) - 0.375 + width*(j+0.5)
            gain, low, high = np.array([tables[s][a][ref] for s in TARGETS]).T
            excludes = (low > 0) | (high < 0)
            ax.vlines(x, low, high, color=COLORS[a], linewidth=2)
            ax.scatter(x[excludes], gain[excludes], s=36, color=COLORS[a], zorder=3, edgecolor="white", linewidth=1.2)
            ax.scatter(x[~excludes], gain[~excludes], s=36, facecolor="white", edgecolor=COLORS[a], linewidth=1.6, zorder=3)
            ax.scatter([], [], s=36, color=COLORS[a], label=LABELS[a])
        style(ax, title)
        ax.set_xticks(range(len(TARGETS)), [NAMES[s] for s in TARGETS])
    axes[0].set_ylabel("Δ Dice (95% interval)", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED))
    labels.append("open marker: interval includes 0")
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dice_fpr_figure(path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, s in zip(axes, ("polypgen", "far")):
        trajectories, _ = load(s)
        for a in ("source_only", "continue", *MAIN):
            picks = [pick(v) for k, v in trajectories.items() if k[0] == a]
            fpr, dice = 100*np.array([p["fpr_any"] for p in picks]), np.array([p["dice"] for p in picks])
            color, marker = COLORS.get(a, SOURCE), "s" if a == "source_only" else "o"
            ax.scatter(fpr, dice, s=14, color=color, alpha=0.35, linewidth=0, marker=marker)
            ax.scatter(fpr.mean(), dice.mean(), s=80, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                       marker=marker, label=LABELS[a])
            # DACS sits next to MinEnt+MMD on PolypGen: label it below-left
            offset, align = ((-7, -12), "right") if a == "dacs" else ((6, 5), "left")
            ax.annotate(LABELS[a], (fpr.mean(), dice.mean()), xytext=offset, textcoords="offset points",
                        fontsize=7.5, color=INK, ha=align)
        style(ax, f"{NAMES[s]}: deployed Dice vs normal-image FPR")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("normal images with any predicted pixel (%)", color=MUTED, fontsize=8)
    axes[0].set_ylabel("deployed Dice (polyp images)", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    gains_figure(FIGURES/"gains.png")
    dice_fpr_figure(FIGURES/"dice_fpr.png")
    print("wrote", FIGURES/"gains.png", FIGURES/"dice_fpr.png")


if __name__ == "__main__":
    main()
