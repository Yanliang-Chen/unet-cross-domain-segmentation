"""Summary figures for the report, from tables already in results/benchmark/.

Usage:
  python -m benchmark.summary_figures
gains.png      paired Dice gain (95% interval) of each method over SourceOnly, the continued-training
               control and source-only continuation, per target (paired.md, controls.md)
dice_fpr.png   deployed Dice against normal-image FPR on PolypGen and FAR: one small dot per
               trajectory (Source-Risk pick), one large marker per method mean, from candidates.csv
controls.png   what the control adds: target images in BatchNorm, and more source training (controls.md)
ablation.png   DACS minus DACS without the cross-domain paste, lambda 0.5 (ablations.md)
holdout.png    the frozen-rule holdout on PolypDB Hanoi (holdout_polypdb.md)
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from benchmark.curves import COLORS, FIGURES, GRID, INK, LABELS, MAIN, MUTED, NAMES, TARGETS, load, pick, style
from benchmark.report import RESULTS

INTERVAL = re.compile(r"([+-]\d+\.\d+) \[([+-]\d+\.\d+), ([+-]\d+\.\d+)\]")
SOURCE = "#3d3c39"
CONT_SOURCE = "#b0aea7"
COLORS = {**COLORS, "continue_source": CONT_SOURCE}
LABELS = {**LABELS, "continue_source": "Source-only continuation"}
NORMALS = ("polypgen", "far")


def paired(scenario):
    """algorithm -> {"control": (gain, low, high), "source": (...)} from paired.md."""
    out = {}
    for line in (RESULTS/scenario/"paired.md").read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        found = INTERVAL.findall(line)
        if len(found) == 2:
            out[cells[0]] = {name: tuple(map(float, f)) for name, f in zip(("control", "source"), found)}
    return out


def sections(name):
    """{section first word: {row label: [intervals in column order]}} from a results table file."""
    out, current = {}, None
    for line in (RESULTS/name).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].split()[0]
            out[current] = {}
        elif current and line.startswith("|"):
            found = INTERVAL.findall(line)
            if found:
                out[current][line.strip("|").split("|")[0].strip()] = [tuple(map(float, f)) for f in found]
    return out


def points(ax, x, values, color, size=36):
    """Interval bars with filled markers where the interval excludes zero, open markers otherwise."""
    gain, low, high = np.array(values).T
    excludes = (low > 0) | (high < 0)
    ax.vlines(x, low, high, color=color, linewidth=2)
    ax.scatter(x[excludes], gain[excludes], s=size, color=color, zorder=3, edgecolor="white", linewidth=1.2)
    ax.scatter(x[~excludes], gain[~excludes], s=size, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3)


def open_marker_legend(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=MUTED))
    labels.append("open marker: interval includes 0")
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gains_figure(path):
    tables = {s: paired(s) for s in TARGETS}
    controls = sections("controls.md")
    reference = {s: {a: controls[NAMES[s]][f"{LABELS[a]} − source-only continuation"][0] for a in MAIN}
                 for s in TARGETS}
    panels = (("Gain over SourceOnly", (*MAIN, "continue", "continue_source"), lambda s, a: tables[s][a]["source"]),
              ("Gain over continued-training control", MAIN, lambda s, a: tables[s][a]["control"]),
              ("Gain over source-only continuation", MAIN, lambda s, a: reference[s][a]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), sharey=True)
    for ax, (title, algorithms, value) in zip(axes, panels):
        ax.axhline(0, color=MUTED, linewidth=1)
        width = 0.8/len(algorithms)
        for j, a in enumerate(algorithms):
            x = np.arange(len(TARGETS)) - 0.4 + width*(j+0.5)
            points(ax, x, [value(s, a) for s in TARGETS], COLORS[a])
            ax.scatter([], [], s=36, color=COLORS[a], label=LABELS[a])
        style(ax, title)
        ax.set_xticks(range(len(TARGETS)), [NAMES[s] for s in TARGETS])
    axes[0].set_ylabel("Δ Dice (95% interval)", color=MUTED, fontsize=8)
    open_marker_legend(fig, axes[0], 6)
    save(fig, path)


def dice_fpr_figure(path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, s in zip(axes, NORMALS):
        trajectories, _ = load(s)
        for a in ("source_only", "continue_source", "continue", *MAIN):
            picks = [pick(v) for k, v in trajectories.items() if k[0] == a]
            fpr, dice = 100*np.array([p["fpr_any"] for p in picks]), np.array([p["dice"] for p in picks])
            color = COLORS.get(a, SOURCE)
            marker = {"source_only": "s", "continue_source": "D"}.get(a, "o")
            ax.scatter(fpr, dice, s=14, color=color, alpha=0.35, linewidth=0, marker=marker)
            ax.scatter(fpr.mean(), dice.mean(), s=80, color=color, edgecolor="white", linewidth=1.5, zorder=3,
                       marker=marker, label=LABELS[a])
            # labels that would collide with a neighbour are moved
            moved = {"dacs": ((-7, -12), "right"),
                     "continue_source": ((6, -13), "left") if s == "polypgen" else ((-8, 4), "right")}
            offset, align = moved.get(a, ((6, 5), "left"))
            ax.annotate(LABELS[a], (fpr.mean(), dice.mean()), xytext=offset, textcoords="offset points",
                        fontsize=7.5, color=INK, ha=align)
        style(ax, f"{NAMES[s]}: deployed Dice vs normal-image FPR")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_xlabel("normal images with any predicted pixel (%)", color=MUTED, fontsize=8)
    axes[0].set_ylabel("deployed Dice (polyp images)", color=MUTED, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.05))
    save(fig, path)


def controls_figure(path):
    controls = sections("controls.md")
    rows = (("Control − source-only continuation (target images in BatchNorm)", "#8a8984",
             "Control − source-only continuation: target images in BatchNorm"),
            ("Source-only continuation − SourceOnly (more source training)", SOURCE,
             "Source-only continuation − SourceOnly: more source training"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [5, 2]})
    for j, (row, color, label) in enumerate(rows):
        x = np.arange(len(TARGETS)) + (j-0.5)*0.3
        points(axes[0], x, [controls[NAMES[s]][row][0] for s in TARGETS], color)
        axes[0].scatter([], [], s=36, color=color, label=label)
        x = np.arange(len(NORMALS)) + (j-0.5)*0.3
        points(axes[1], x, [controls[NAMES[s]][row][2] for s in NORMALS], color)
    for ax, title, targets, unit in ((axes[0], "Δ Dice", TARGETS, "Δ Dice (95% interval)"),
                                     (axes[1], "Δ normal-image FPR", NORMALS, "Δ FPR (points)")):
        ax.axhline(0, color=MUTED, linewidth=1)
        style(ax, title)
        ax.set_xticks(range(len(targets)), [NAMES[s] for s in targets])
        ax.set_ylabel(unit, color=MUTED, fontsize=8)
    open_marker_legend(fig, axes[0], 3)
    save(fig, path)


def ablation_figure(path):
    ablation = sections("ablations.md")
    row = "DACS − DACS without paste"
    color = COLORS["dacs"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"width_ratios": [5, 5, 2]})
    panels = ((axes[0], "Δ Dice", TARGETS, 0, "Δ Dice (95% interval)"),
              (axes[1], "Δ small-polyp Dice", TARGETS, 1, "Δ small-polyp Dice"),
              (axes[2], "Δ normal-image FPR", NORMALS, 2, "Δ FPR (points)"))
    for ax, title, targets, column, unit in panels:
        ax.axhline(0, color=MUTED, linewidth=1)
        points(ax, np.arange(len(targets)), [ablation[NAMES[s]][row][column] for s in targets], color)
        style(ax, title)
        ax.set_xticks(range(len(targets)), [NAMES[s] for s in targets])
        ax.set_xlim(-0.5, len(targets)-0.5)
        ax.set_ylabel(unit, color=MUTED, fontsize=8)
    axes[0].scatter([], [], s=36, color=color,
                    label="DACS − DACS without paste (λ 0.5): positive Dice = the paste helps; positive FPR = the paste adds false positives")
    open_marker_legend(fig, axes[0], 2)
    save(fig, path)


def holdout_figure(path):
    lines = (RESULTS/"holdout_polypdb.md").read_text(encoding="utf-8").splitlines()
    primary = next(tuple(map(float, INTERVAL.search(l).groups())) for l in lines if l.startswith("DACS λ=1, Source-Risk"))
    rows = [("DACS λ=1 − SourceOnly (primary, frozen)", primary, COLORS["dacs"])]
    for l in lines:
        if l.startswith("| ") and " − " in l and INTERVAL.search(l) and not l.startswith("| Primary"):
            label = l.strip("|").split("|")[0].strip()
            color = CONT_SOURCE if label.startswith("Source-only") else COLORS["dacs"]
            rows.append((label, tuple(map(float, INTERVAL.search(l).groups())), color))
    fig, ax = plt.subplots(figsize=(8, 0.5*len(rows)+1.2))
    ax.axvline(0, color=MUTED, linewidth=1)
    for i, (label, (g, lo, hi), color) in enumerate(rows):
        y = len(rows)-1-i
        excludes = lo > 0 or hi < 0
        ax.hlines(y, lo, hi, color=color, linewidth=2)
        ax.scatter([g], [y], s=40, zorder=3, color=color if excludes else "white", edgecolor=color if not excludes else "white",
                   linewidth=1.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1], fontsize=8, color=INK)
    style(ax, "PolypDB Hanoi holdout (1,000 images, 256 patients): paired Δ Dice, 95% interval")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ Dice", color=MUTED, fontsize=8)
    save(fig, path)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, draw in (("gains", gains_figure), ("dice_fpr", dice_fpr_figure), ("controls", controls_figure),
                       ("ablation", ablation_figure), ("holdout", holdout_figure)):
        draw(FIGURES/f"{name}.png")
        print("wrote", FIGURES/f"{name}.png")


if __name__ == "__main__":
    main()
