"""Report tables kept in git (runs/ is ignored): per-candidate scores and the paired gain of
every method over the continued-training control and over SourceOnly.

Usage:
  python -m benchmark.report --scenario clinicdb
Writes results/benchmark/<scenario>/{candidates.csv, paired.md} and copies the selection
analyses runs/benchmark/analysis/<scenario>_{full,methods}.md next to them.
"""
import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

from benchmark.analyze import OUT, VALIDATORS, load_pools

RESULTS = Path("results/benchmark")
POOLS = ("source_only", "uda_adabn", "uda_minent", "uda_advent", "uda_dann", "uda_mmd", "uda_bnm",
         "uda_mcc", "uda_meanteacher", "uda_fda", "uda_minent_mmd", "uda_dacs", "uda_continue")
METRICS = ("dice", "group_dice", "iou", "small_dice", "fpr_any", "fpr_area", "auroc")
BOOTSTRAP, SEED = 2000, 20261101


def base_run(row):
    return re.search(r"a[01]_seed\d+", row["run"]).group(0)


def trajectories(rows):
    """(algorithm, base run, lambda) -> candidates; one saved training trajectory each."""
    out = defaultdict(list)
    for r in rows:
        out[(r["algorithm"], base_run(r), r["lam"])].append(r)
    return out


def write_candidates(rows, path):
    fields = ["id", "pool", "algorithm", "lam", "augmentation", "seed", "step", "base_run",
              *METRICS, *VALIDATORS, "foreground_fraction"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **r["validators"], "base_run": base_run(r)})


def paired(rows, units):
    """Each trajectory deploys its Source-Risk pick; compare with the same base run's control
    (and SourceOnly pick). Interval: bootstrap over base runs and image groups jointly."""
    picks = {k: max(v, key=lambda r: r["validators"]["source_risk"]) for k, v in trajectories(rows).items()}
    control = {b: r for (a, b, l), r in picks.items() if a == "continue"}
    source = {b: r for (a, b, l), r in picks.items() if a == "source_only"}
    labels, inverse = np.unique(units, return_inverse=True) if units.size and all(units) else \
        (np.arange(len(units)), np.arange(len(units)))
    rng, table = np.random.default_rng(SEED), []
    for algorithm in sorted({a for a, _, _ in picks}):
        keys = [k for k in picks if k[0] == algorithm]
        runs = sorted({b for _, b, _ in keys})
        row = {"algorithm": algorithm, "n": len(keys),
               "dice": float(np.mean([picks[k]["dice"] for k in keys])),
               "fpr": f"{np.mean([picks[k]['fpr_any'] for k in keys]):.3f}" if "fpr_any" in picks[keys[0]] else "–"}
        for name, ref in (("control", control), ("source", source)):
            # per base run: mean over its lambdas of per-group Dice differences
            per_run = np.stack([np.mean([np.bincount(inverse, picks[k]["_dice"]-ref[b]["_dice"], len(labels))
                                         / np.bincount(inverse, minlength=len(labels))
                                         for k in keys if k[1] == b], 0) for b in runs])
            draws = per_run[rng.integers(0, len(runs), (BOOTSTRAP, len(runs)))]
            groups = rng.integers(0, len(labels), (BOOTSTRAP, len(labels)))
            means = np.take_along_axis(draws, groups[:, None, :], 2).mean((1, 2))
            row[name] = float(per_run.mean())
            row[f"{name}_ci"] = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))
            row[f"{name}_wins"] = sum(picks[k]["dice"] > ref[k[1]]["dice"] for k in keys)
        table.append(row)
    return sorted(table, key=lambda r: -r["dice"])


def markdown_paired(table, scenario):
    lines = [f"# Paired gains on {scenario}: Source-Risk pick per trajectory",
             "", "Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; "
             f"95% interval from {BOOTSTRAP} joint bootstrap draws over base runs and image groups.", "",
             "| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |",
             "|---|---:|---:|---:|---|---:|---|---:|"]
    for r in table:
        c, s = r["control_ci"], r["source_ci"]
        lines.append(f"| {r['algorithm']} | {r['n']} | {r['dice']:.3f} | {r['fpr']} | {r['control']:+.3f} "
                     f"[{c[0]:+.3f}, {c[1]:+.3f}] | {r['control_wins']}/{r['n']} | {r['source']:+.3f} "
                     f"[{s[0]:+.3f}, {s[1]:+.3f}] | {r['source_wins']}/{r['n']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()
    table, units = load_pools(POOLS, features=False)
    rows = table[args.scenario]
    out = RESULTS/args.scenario
    out.mkdir(parents=True, exist_ok=True)
    write_candidates(rows, out/"candidates.csv")
    (out/"paired.md").write_text(markdown_paired(paired(rows, units[args.scenario]), args.scenario), encoding="utf-8")
    for name in ("full", "methods"):
        src = OUT/f"{args.scenario}_{name}.md"
        if src.exists():
            shutil.copy(src, out/src.name)
    print("wrote", out)


if __name__ == "__main__":
    main()
