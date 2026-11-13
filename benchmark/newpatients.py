"""New-patient test of H3 on FAR (benchmark_protocol.yaml: amendment_v2.new_patient_test).

Usage:
  python -m benchmark.newpatients
Adaptation uses the unlabelled images of half the FAR patients; Dice and normal-image FPR are
measured on the other half, never seen during adaptation. Writes results/benchmark/new_patients_far.md.
Run once, after the rule was committed.
"""
from benchmark.analyze import load_pools
from benchmark.compare import Target, interval
from benchmark.report import RESULTS

SCENARIO = "far_split"
POOLS = ("source_only", "uda_continue_source", "uda_dacs", "uda_dacs_nopaste")
REFERENCE = "continue_source"
PRIMARY = (("dacs", 1.0, "DACS λ=1 (deployment rule)"), ("dacs_nopaste", 0.5, "DACS without paste λ=0.5"))
MODELS = (("source_only", None, "SourceOnly"), ("continue_source", None, "Source-only continuation"),
          ("dacs", 1.0, "DACS λ=1"), ("dacs", 0.5, "DACS λ=0.5"), ("dacs_nopaste", 0.5, "DACS without paste λ=0.5"))
FPR_MARGIN = 0.05            # acceptable increase in normal-image FPR over the reference (5 points)


def main():
    table, units = load_pools(POOLS, features=False)
    t = Target(SCENARIO, table[SCENARIO], units[SCENARIO])
    sets = t.sets
    lines = ["# New-patient test on FAR (amendment v2)", "",
             f"Test patients never used for adaptation: {int(sets['polyp'].sum())} polyp images and "
             f"{int(sets['normal'].sum())} normal images; adaptation used the unlabelled images of the other "
             "patients. **Levels** are image means over the test images (mean of 6 base runs). **Differences** "
             "are paired by base run and averaged per patient group, with 95% intervals from a joint bootstrap "
             "over base runs and patient groups. Groups resampled per endpoint: Dice, the 93 test patients with "
             "polyp images; small-polyp Dice, 47; normal-image FPR, all 222 test patients (those with normal "
             "images). Rule frozen in a commit before any run; evaluated once.", "",
             "## Pre-set criteria (reference: source-only continuation)", "",
             "| Model | Δ Dice [95% CI] | Dice criterion (lower bound > 0) | Δ normal FPR, points [95% CI] | FPR criterion (upper bound ≤ +5) | H3 on new patients |",
             "|---|---|---|---|---|---|"]
    for algorithm, lam, label in PRIMARY:
        d = t.gain(algorithm, REFERENCE, "dice", lam)
        f = t.gain(algorithm, REFERENCE, "fpr_any", lam)
        dice_ok, fpr_ok = d[1] > 0, f[2] <= FPR_MARGIN
        lines.append(f"| {label} | {interval(d)} | {'met' if dice_ok else 'not met'} | {interval(f, 100, 1)} | "
                     f"{'met' if fpr_ok else 'not met'} | **{'supported' if dice_ok and fpr_ok else 'not supported'}** |")
    lines += ["", "## Levels on the test patients (image means)", "",
              "| Model | Dice | Small-polyp Dice | Normal FPR (any pixel) | Normal FPR (area ≥ 0.1%) |", "|---|---:|---:|---:|---:|"]
    for algorithm, lam, label in MODELS:
        if t.keys(algorithm, lam):
            lines.append(f"| {label} | {t.level(algorithm, 'dice', lam):.3f} | {t.level(algorithm, 'small', lam):.3f} | "
                         f"{100*t.level(algorithm, 'fpr_any', lam):.1f}% | {100*t.level(algorithm, 'fpr_area', lam):.1f}% |")
    lines += ["", "## Secondary paired differences (declared before running; patient groups)", "",
              "| Comparison | Δ Dice | Δ small-polyp Dice | Δ normal FPR (points) | Δ area FPR (points) |", "|---|---|---|---|---|"]
    rows = [("dacs", 1.0, "source_only", "DACS λ=1 − SourceOnly"),
            ("dacs_nopaste", 0.5, "source_only", "DACS without paste λ=0.5 − SourceOnly"),
            ("continue_source", None, "source_only", "Source-only continuation − SourceOnly"),
            ("dacs", 0.5, "continue_source", "DACS λ=0.5 − source-only continuation"),
            ("dacs", 0.5, "dacs_nopaste", "DACS λ=0.5 − DACS without paste λ=0.5 (effect of the paste)")]
    for algorithm, lam, ref, label in rows:
        if t.keys(algorithm, lam) and t.keys(ref):
            lines.append(f"| {label} | {interval(t.gain(algorithm, ref, 'dice', lam))} | "
                         f"{interval(t.gain(algorithm, ref, 'small', lam))} | "
                         f"{interval(t.gain(algorithm, ref, 'fpr_any', lam), 100, 1)} | "
                         f"{interval(t.gain(algorithm, ref, 'fpr_area', lam), 100, 1)} |")
    (RESULTS/"new_patients_far.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("wrote", RESULTS/"new_patients_far.md")


if __name__ == "__main__":
    main()
