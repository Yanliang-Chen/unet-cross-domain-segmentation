"""Phase 3: one-time evaluation of the frozen rule on the PolypDB Hanoi holdout
(benchmark_protocol.yaml: amendment_v2.phase3_holdout).

Usage:
  python -m benchmark.holdout
Writes results/benchmark/holdout_polypdb.md. Run once, after the rule was committed; the
protocol forbids changing the rule or rerunning with other settings after seeing this table.
"""
import numpy as np

from benchmark.analyze import load_pools
from benchmark.compare import Target, interval
from benchmark.report import RESULTS, base_run

SCENARIO = "polypdb"
POOLS = ("source_only", "uda_continue_source", "uda_dacs", "uda_dacs_nopaste")
PRIMARY = ("dacs", 1.0)
SECONDARY = [("dacs", 1.0, "continue_source"), ("dacs_nopaste", 0.5, "source_only"),
             ("dacs_nopaste", 0.5, "continue_source"), ("continue_source", None, "source_only")]
LABELS = {"dacs": "DACS λ=1", "dacs_nopaste": "DACS without paste λ=0.5", "continue_source": "Source-only continuation",
          "source_only": "SourceOnly"}


def single(t, algorithm, lam=None):
    picks = [t.picks[k] for k in t.keys(algorithm, lam)]
    return max(picks, key=lambda r: r["validators"]["source_risk"])


def main():
    table, units = load_pools(POOLS, features=False)
    t = Target(SCENARIO, table[SCENARIO], units[SCENARIO])
    method, lam = PRIMARY
    g = t.gain(method, "source_only", "dice", lam)
    confirmed = g[1] > 0
    lines = ["# Phase 3: frozen-rule holdout on PolypDB Hanoi (amendment v2)", "",
             f"{int(t.sets['polyp'].sum())} WLI polyp images from Hanoi Medical University / IGH (Fujinon), never used "
             "before; transductive like the other targets; bootstrap over the 256 patient groups. Rule and analysis "
             "fixed in a commit before any prediction on these images; evaluated once.", "",
             "## Primary test", "",
             f"DACS λ=1, Source-Risk pick per trajectory, paired Dice gain over the same base run's SourceOnly pick: "
             f"**{interval(g)}** ({g[3]}/{g[4]} runs better) → **{'confirmed' if confirmed else 'not confirmed'}** "
             "(criterion: interval above zero).", "",
             "## Levels", "",
             "| Model | Mean Dice (6 runs) | Small-polyp Dice | Single deployed model (run, step): Dice |",
             "|---|---:|---:|---|"]
    for a, l in (("source_only", None), ("continue_source", None), ("dacs", 1.0), ("dacs_nopaste", 0.5)):
        if not t.keys(a, l):
            continue
        s = single(t, a, l)
        lines.append(f"| {LABELS[a]} | {t.level(a, 'dice', l):.3f} | {t.level(a, 'small', l):.3f} | "
                     f"{base_run(s)} step {s['step']}: {t.measure(s, 'dice').mean():.3f} |")
    lines += ["", "## Secondary comparisons (declared before running)", "",
              "| Comparison | Dice [95% CI] | Small-polyp Dice [95% CI] |", "|---|---|---|"]
    for a, l, ref in SECONDARY:
        if t.keys(a, l) and t.keys(ref):
            lines.append(f"| {LABELS[a]} − {LABELS[ref]} | {interval(t.gain(a, ref, 'dice', l))} | "
                         f"{interval(t.gain(a, ref, 'small', l))} |")
    lines.append(f"| Primary small-polyp gain (DACS λ=1 − SourceOnly) | – | {interval(t.gain(method, 'source_only', 'small', lam))} |")
    (RESULTS/"holdout_polypdb.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("wrote", RESULTS/"holdout_polypdb.md", "primary", "confirmed" if confirmed else "not confirmed")


if __name__ == "__main__":
    main()
