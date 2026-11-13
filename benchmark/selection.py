"""Unified label-free selection table (benchmark_protocol.yaml: amendment_v2.selection_table).

Usage:
  python -m benchmark.analyze --pools source_only --name v2_source_only
  python -m benchmark.analyze --pools <all 13 pools> --name v2_all
  python -m benchmark.selection
Reads runs/benchmark/analysis/{v2_source_only,v2_all}.json; writes results/benchmark/selection.md.
One row per target and pool: methods, candidates, validators scored for every candidate, Oracle,
the Source-Risk pick and its loss, and the best and worst other validators.
"""
import json
from benchmark.analyze import OUT, VALIDATORS
from benchmark.report import RESULTS, UNITS

TARGETS = ("clinicdb", "etis", "colondb", "polypgen", "far")
NAMES = {"clinicdb": "ClinicDB", "etis": "ETIS", "colondb": "ColonDB", "polypgen": "PolypGen", "far": "FAR"}
POOLS = {"v2_source_only": "SourceOnly only", "v2_all": "all scored methods"}


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
    lines = ["# Label-free model selection over whole candidate pools (amendment v2)", "",
             "Positive-polyp Dice (v2 metrics). Each pool holds every saved checkpoint of the listed methods on "
             "that target (all lambdas and base runs). Oracle = best candidate by target labels. Loss = Oracle "
             "Dice − picked Dice, 95% interval from the group bootstrap in analyze.py. Rank = Source-Risk's "
             "position among the validators scored for every candidate (1 = smallest loss). This is selection "
             "across methods; the main results instead fix the method and use Source-Risk only to pick the "
             "checkpoint within each trajectory. `clinicdb/clinicdb_full.md` is the v1 analysis of ClinicDB's "
             "1,860-candidate pool, the only adapted pool with all 19 validators (ClinicDB has no empty masks, "
             "so v1 = v2 there). Oracle and pick Dice are image means; the loss interval resamples groups "
             "only: " + "; ".join(f"{NAMES[t]}: {UNITS[t]}" for t in TARGETS) + ".", "",
             "| Target | Pool | Methods | Candidates | Validators scored | Oracle Dice | Source-Risk pick | Source-Risk loss [95% CI] | Source-Risk rank | Best other validator | Worst validator |",
             "|---|---|---:|---:|---|---|---|---|---:|---|---|"]
    for s in TARGETS:
        for name, label in POOLS.items():
            lines.append(row(s, label, reports[name]))
    (RESULTS/"selection.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("wrote", RESULTS/"selection.md")


if __name__ == "__main__":
    main()
