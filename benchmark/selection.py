"""Label-free checkpoint selection (H4) (benchmark_protocol.yaml: amendment_v2.selection_table).

Usage:
  python -m benchmark.analyze --pools source_only --name v2_source_only
  python -m benchmark.analyze --pools <the 13 original pools> --name v2_all
  python -m benchmark.analyze --pools <all 16 pools in report.POOLS> --name v2_full
  python -m benchmark.selection
Reads runs/benchmark/analysis/{v2_source_only,v2_all,v2_full}.json and results/benchmark/<target>/
candidates.csv; writes results/benchmark/selection.md.
Part 1, the deployment rule: Source-Risk picks one checkpoint inside each trajectory; how far below the
trajectory's best saved checkpoint it lands, per target and method, and where it fails.
Part 2, selection across methods: one row per target and pool (SourceOnly only; the original protocol
pool of 13 methods; the complete pool with the later controls and ablations), with Oracle, the
Source-Risk pick and its loss, and the best and worst other validators.
"""
import json
import numpy as np
from benchmark.analyze import OUT, VALIDATORS
from benchmark.curves import LABELS, load, pick
from benchmark.report import RESULTS, UNITS

TARGETS = ("clinicdb", "etis", "colondb", "polypgen", "far")
NAMES = {"clinicdb": "ClinicDB", "etis": "ETIS", "colondb": "ColonDB", "polypgen": "PolypGen", "far": "FAR"}
POOLS = {"v2_source_only": "SourceOnly only", "v2_all": "original protocol pool",
         "v2_full": "complete pool (+ later controls and ablations)"}
LABELS = {**LABELS, "continue_source": "Source-only continuation", "dacs_nopaste": "DACS without paste",
          "dacs_noweight": "DACS without confidence weight"}
ORDER = ("dacs", "dacs_nopaste", "dacs_noweight", "minent_mmd", "fda", "minent", "mmd", "dann", "advent", "bnm",
         "mcc", "meanteacher", "adabn", "continue_source", "continue", "source_only")
FAIL = 0.05     # a pick this far below the trajectory's best saved checkpoint counts as a failure


def within(scenario):
    """Per method: trajectories, mean pick and best Dice, mean and max regret, failures."""
    trajectories, _ = load(scenario)
    out = {}
    for a in ORDER:
        runs = [v for k, v in trajectories.items() if k[0] == a]
        if not runs:
            continue
        chosen = np.array([pick(v)["dice"] for v in runs])
        best = np.array([max(r["dice"] for r in v) for v in runs])
        regret = best - chosen
        out[a] = (len(runs), chosen.mean(), best.mean(), regret.mean(), regret.max(), int((regret >= FAIL).sum()))
    return out


def row(scenario, label, report):
    pools = report["scenarios"][scenario]
    whole = pools["across_all"]
    methods = sorted(k.split("=", 1)[1] for k in pools if k.startswith("algorithm=")) or ["source_only"]
    validators = whole["validators"]
    missing = [v for v in VALIDATORS if v not in validators]
    risk = validators["source_risk"]
    others = sorted(((v, e) for v, e in validators.items() if v != "source_risk"), key=lambda x: x[1]["gap"])
    rank = 1 + sum(e["gap"] < risk["gap"] for _, e in others)
    ci = risk.get("gap_ci95")
    best, worst = others[0], others[-1]
    return (f"| {NAMES[scenario]} | {label} | {len(methods)} | {whole['n']} | {len(validators)}/{len(VALIDATORS)}"
            + (f" (missing: {', '.join(missing)})" if missing else "")
            + f" | {whole['oracle']['dice']:.3f} ({whole['oracle']['algorithm']}) | "
            f"{risk['pick']['algorithm']} {risk['pick']['dice']:.3f} | {risk['gap']:.3f}"
            + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f" | {rank}/{len(validators)} | "
            f"{best[0]}: {best[1]['pick']['algorithm']} {best[1]['pick']['dice']:.3f} (loss {best[1]['gap']:.3f}) | "
            f"{worst[0]}: {worst[1]['pick']['algorithm']} {worst[1]['pick']['dice']:.3f} (loss {worst[1]['gap']:.3f}) |")


def main():
    reports = {name: json.loads((OUT/f"{name}.json").read_text(encoding="utf-8")) for name in POOLS}
    lines = ["# Label-free checkpoint selection (H4, amendment v2)", "",
             "Positive-polyp Dice (v2 metrics), image means. H4 is a supporting analysis: it explains how the "
             "deployed checkpoint is chosen without target labels and where that choice fails.", "",
             "## 1. The deployment rule: Source-Risk inside each trajectory", "",
             "Each trajectory is one method × base run × λ with its saved checkpoints (SourceOnly: the 30 epochs of a "
             "source run). Pick = the checkpoint with the highest source-validation Dice; best = the best saved "
             "checkpoint by target Dice (needs labels); regret = best − pick. Failures = trajectories with regret "
             f"≥ {FAIL} Dice, a descriptive threshold set after the curves analysis (about the size of a typical "
             "method gain), not a test. Checkpoint spacing differs by target (every 200 steps on ClinicDB, ETIS, "
             "ColonDB; every 400 for the methods on PolypGen and FAR), so regret is not comparable across "
             "spacings.", "",
             "| Target | Method | Trajectories | Pick Dice | Best Dice | Mean regret | Max regret | Failures |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for s in TARGETS:
        for a, (n, chosen, best, regret, worst, fails) in within(s).items():
            lines.append(f"| {NAMES[s]} | {LABELS[a]} | {n} | {chosen:.3f} | {best:.3f} | {regret:.3f} | "
                         f"{worst:.3f} | {fails}/{n} |")
    lines += ["", "## 2. Selection across methods (not used for deployment)", "",
              "Each pool holds every saved checkpoint of the listed methods on that target (all lambdas and base "
              "runs). **Original protocol pool** = the 13 pools of the original protocol (SourceOnly, 11 methods, "
              "the continued-training control) restricted to those run on the target. **Complete pool** = the "
              "original pool plus the later source-only continuation and DACS ablations (no paste; no confidence "
              "weight on ClinicDB and FAR). Oracle = best candidate by target labels. Loss = Oracle Dice − picked "
              "Dice, 95% interval from the group bootstrap in analyze.py. Rank = Source-Risk's position among the "
              "validators scored for every candidate (1 = smallest loss). `clinicdb/clinicdb_full.md` is the v1 "
              "analysis of ClinicDB's 1,860-candidate pool, the only adapted pool with all 19 validators "
              "(ClinicDB has no empty masks, so v1 = v2 there). The loss interval resamples the Dice groups: "
              + "; ".join(f"{NAMES[t]}: {UNITS[t]}" for t in TARGETS) + ".", "",
              "| Target | Pool | Methods | Candidates | Validators scored | Oracle Dice | Source-Risk pick | Source-Risk loss [95% CI] | Source-Risk rank | Best other validator | Worst validator |",
              "|---|---|---:|---:|---|---|---|---|---:|---|---|"]
    for s in TARGETS:
        for name, label in POOLS.items():
            lines.append(row(s, label, reports[name]))
    (RESULTS/"selection.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("wrote", RESULTS/"selection.md")


if __name__ == "__main__":
    main()
