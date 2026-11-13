"""Amendment v2 analyses of the deployed models (benchmark_protocol.yaml: amendment_v2).

Usage:
  python -m benchmark.compare
Each trajectory deploys its Source-Risk pick, as in report.py. Writes to results/benchmark/:
  h2.md          direct paired comparisons MinEnt+MMD vs MinEnt, MMD; DACS vs MinEnt, MMD, FDA
  endpoints.md   per target and method vs SourceOnly: Dice, small-polyp Dice, normal FPR (any pixel,
                 area >= 0.1%), empty-mask FPR, with paired intervals and a verdict
  deployment.md  the post hoc deployment rule (method and lambda from ClinicDB) on every target
  controls.md    phase 2: source-only continued training against the control, SourceOnly and methods
  ablations.md   phase 2: DACS against DACS without paste / without confidence weight (lambda 0.5)
Pairing: same base run and lambda (a lambda-0 reference pairs with every lambda of its base run);
per-group differences averaged over lambdas per base run; joint bootstrap over base runs and groups.
"""
import numpy as np

from benchmark.analyze import image_sets, load_pools
from benchmark.report import BOOTSTRAP, POOLS, RESULTS, SEED, base_run, paired, trajectories
from benchmark.score import SCORES

TARGETS = ("clinicdb", "etis", "colondb", "polypgen", "far")
NAMES = {"clinicdb": "ClinicDB", "etis": "ETIS", "colondb": "ColonDB", "polypgen": "PolypGen", "far": "FAR"}
LABELS = {"dacs": "DACS", "minent_mmd": "MinEnt+MMD", "fda": "FDA", "minent": "MinEnt", "mmd": "MMD",
          "dann": "DANN", "continue": "Control", "source_only": "SourceOnly", "adabn": "AdaBN",
          "continue_source": "Source-only continuation", "dacs_nopaste": "DACS without paste",
          "dacs_noweight": "DACS without confidence weight"}
H2 = {"minent_mmd": ("minent", "mmd"), "dacs": ("minent", "mmd", "fda")}
H2_TARGETS = ("clinicdb", "etis", "colondb")
METHODS = ("dacs", "minent_mmd", "fda", "minent", "mmd", "dann", "continue", "continue_source", "adabn")
ABLATION_TARGETS, ABLATIONS = ("clinicdb", "etis", "colondb", "polypgen", "far"), ("dacs_nopaste", "dacs_noweight")
RULE_METHODS, RULE_LAMS = ("dacs", "minent_mmd", "fda"), (0.1, 0.5, 1.0)


class Target:
    """Deployed picks of one target with their per-image scores."""

    def __init__(self, scenario, rows, units):
        self.scenario, self.rows = scenario, rows
        self.picks = {k: max(v, key=lambda r: r["validators"]["source_risk"]) for k, v in trajectories(rows).items()}
        self.meta = np.load(SCORES/f"meta_{scenario}.npz")
        self.units = units
        self.scores = {}
        first = next(iter(self.picks.values()))
        self.sets = image_sets(self.meta, self.load(first)["is_test"])

    def load(self, row):
        if row["id"] not in self.scores:
            data = np.load(SCORES/row["pool"]/self.scenario/f"{row['id']}.npz")
            self.scores[row["id"]] = {k: data[k] for k in ("dice", "fg_native", "is_test")}
        return self.scores[row["id"]]

    def measure(self, row, name):
        """Per-image values of one endpoint on its image set."""
        s, sets = self.load(row), self.sets
        return {"dice": lambda: s["dice"][sets["polyp"]], "small": lambda: s["dice"][sets["small"]],
                "fpr_any": lambda: (s["fg_native"][sets["normal"]] > 0).astype(float),
                "fpr_area": lambda: (s["fg_native"][sets["normal"]] >= 0.001).astype(float),
                "empty": lambda: (s["fg_native"][sets["empty"]] > 0).astype(float)}[name]()

    def groups(self, name):
        mask = self.sets[{"dice": "polyp", "small": "small", "fpr_any": "normal", "fpr_area": "normal",
                          "empty": "empty"}[name]]
        return self.meta["group"][mask]

    def has(self, name):
        return bool(self.sets[{"dice": "polyp", "small": "small", "fpr_any": "normal", "fpr_area": "normal",
                               "empty": "empty"}[name]].any())

    def keys(self, algorithm, lam=None):
        return sorted(k for k in self.picks if k[0] == algorithm and (lam is None or k[2] == lam))

    def partner(self, key, reference):
        """Same base run and lambda; a reference trained only at lambda 0 pairs with every lambda."""
        exact = (reference, key[1], key[2])
        return self.picks.get(exact) or self.picks[(reference, key[1], 0.0)]

    def gain(self, algorithm, reference, name, lam=None, seed=SEED):
        """Mean paired difference (algorithm - reference), 95% interval, trajectories with diff > 0."""
        keys = self.keys(algorithm, lam)
        groups = self.groups(name)
        if groups.size and all(groups):
            labels, inverse = np.unique(groups, return_inverse=True)
        else:
            labels, inverse = np.arange(len(groups)), np.arange(len(groups))
        diffs = {k: self.measure(self.picks[k], name) - self.measure(self.partner(k, reference), name) for k in keys}
        runs = sorted({k[1] for k in keys})
        per_run = np.stack([np.mean([np.bincount(inverse, diffs[k], len(labels))
                                     / np.bincount(inverse, minlength=len(labels))
                                     for k in keys if k[1] == b], 0) for b in runs])
        rng = np.random.default_rng(seed)
        draws = per_run[rng.integers(0, len(runs), (BOOTSTRAP, len(runs)))]
        pick = rng.integers(0, len(labels), (BOOTSTRAP, len(labels)))
        means = np.take_along_axis(draws, pick[:, None, :], 2).mean((1, 2))
        low, high = np.quantile(means, [0.025, 0.975])
        return float(per_run.mean()), float(low), float(high), sum(diffs[k].mean() > 0 for k in keys), len(keys)

    def level(self, algorithm, name, lam=None):
        return float(np.mean([self.measure(self.picks[k], name).mean() for k in self.keys(algorithm, lam)]))


def interval(g, scale=1.0, digits=3):
    return f"{scale*g[0]:+.{digits}f} [{scale*g[1]:+.{digits}f}, {scale*g[2]:+.{digits}f}]"


def h2_table(targets):
    lines = ["# H2: direct paired comparisons (amendment v2)", "",
             "Gain = method − comparator, same base run and lambda, positive-polyp Dice; 95% interval from "
             f"{BOOTSTRAP} joint bootstrap draws over base runs and groups. Wins = trajectories with a "
             "positive mean difference. Supported if the interval excludes zero against every named "
             "comparator on at least 2 of 3 targets.", "",
             "| Method | Comparator | " + " | ".join(NAMES[s] for s in H2_TARGETS) + " |",
             "|---|---|" + "---|"*len(H2_TARGETS)]
    verdicts = []
    for method, comparators in H2.items():
        passed = {s: True for s in H2_TARGETS}
        for c in comparators:
            cells = []
            for s in H2_TARGETS:
                g = targets[s].gain(method, c, "dice")
                passed[s] &= g[1] > 0
                cells.append(f"{interval(g)} ({g[3]}/{g[4]})")
            lines.append(f"| {LABELS[method]} | {LABELS[c]} | " + " | ".join(cells) + " |")
        n = sum(passed.values())
        verdicts.append(f"- **{LABELS[method]}:** interval above zero against every comparator on {n}/3 targets "
                        f"({', '.join(NAMES[s] for s in H2_TARGETS if passed[s]) or 'none'}) → "
                        + ("supported." if n >= 2 else "means support it, direct comparison inconclusive."))
    return lines + ["", *verdicts, ""]


def verdict(dice, fpr):
    better = dice[1] > 0
    worse = dice[2] < 0
    if fpr is None:
        return "Dice better" if better else "Dice worse" if worse else "not enough evidence"
    lower, higher = fpr[2] < 0, fpr[1] > 0
    if better and lower:
        return "both"
    if better and higher:
        return "trade-off (Dice ↑, FPR ↑)"
    if lower and worse:
        return "trade-off (FPR ↓, Dice ↓)"
    if better:
        return "Dice better"
    if lower:
        return "FPR lower"
    if worse and higher:
        return "worse (Dice ↓, FPR ↑)"
    if worse:
        return "Dice worse"
    if higher:
        return "FPR higher"
    return "not enough evidence"


def endpoint_table(targets, paired_tables):
    lines = ["# Endpoints vs SourceOnly (amendment v2)", "",
             "Deployed Source-Risk picks; each cell is the method's mean level, then the paired difference to "
             "the same base run's SourceOnly pick with its 95% interval (joint bootstrap over base runs and "
             "groups). Dice column uses report.py's intervals; the other columns use a fixed seed per "
             "comparison. FPR in percentage points. Verdict per amendment_v2.endpoints: Dice interval above "
             "zero = Dice better; normal-FPR (any pixel) interval below zero = FPR lower. Levels are image means; "
             "differences are group-weighted (video/centre/patient where available), so a difference can "
             "disagree with the gap between two levels when groups differ in size (PolypGen empty-mask "
             "images come from few groups).", ""]
    for s in TARGETS:
        t = targets[s]
        normal, empty = t.has("fpr_any"), t.has("empty")
        head = ["Method", "Dice", "Small-polyp Dice"]
        head += ["Normal FPR (any)", "Normal FPR (area ≥0.1%)"] if normal else []
        head += ["Empty-mask FPR"] if empty else []
        head.append("Verdict")
        n_small = int(t.sets["small"].sum())
        lines += [f"## {NAMES[s]} ({int(t.sets['polyp'].sum())} polyp images, {n_small} small"
                  + (f", {int(t.sets['normal'].sum())} normal" if normal else "")
                  + (f", {int(t.sets['empty'].sum())} empty-mask" if empty else "") + ")", "",
                  "| " + " | ".join(head) + " |", "|---|" + "---|"*(len(head)-1)]
        source = {"dice": t.level("source_only", "dice"), "small": t.level("source_only", "small")}
        cells = [f"SourceOnly", f"{source['dice']:.3f}", f"{source['small']:.3f}"]
        if normal:
            cells += [f"{100*t.level('source_only', 'fpr_any'):.1f}%", f"{100*t.level('source_only', 'fpr_area'):.1f}%"]
        if empty:
            cells.append(f"{100*t.level('source_only', 'empty'):.1f}%")
        lines.append("| " + " | ".join(cells) + " | reference |")
        rows = {r["algorithm"]: r for r in paired_tables[s]}
        for a in METHODS:
            if not t.keys(a):
                continue
            d = (rows[a]["source"], *rows[a]["source_ci"])
            small = t.gain(a, "source_only", "small")
            cells = [LABELS[a], f"{t.level(a, 'dice'):.3f} {interval(d)}", f"{t.level(a, 'small'):.3f} {interval(small)}"]
            fpr = None
            if normal:
                fpr = t.gain(a, "source_only", "fpr_any")
                area = t.gain(a, "source_only", "fpr_area")
                cells += [f"{100*t.level(a, 'fpr_any'):.1f}% {interval(fpr, 100, 1)}",
                          f"{100*t.level(a, 'fpr_area'):.1f}% {interval(area, 100, 1)}"]
            if empty:
                cells.append(f"{100*t.level(a, 'empty'):.1f}% {interval(t.gain(a, 'source_only', 'empty'), 100, 1)}")
            cells.append(verdict(d, fpr))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def deployment(targets):
    clinic = targets["clinicdb"]
    scores = {(a, l): clinic.level(a, "dice", l) for a in RULE_METHODS for l in RULE_LAMS}
    method, lam = max(scores, key=scores.get)
    lines = ["# Deployment rule (amendment v2, post hoc)", "",
             "Rule: (method, λ) with the highest mean deployed Dice on ClinicDB among "
             "{DACS, MinEnt+MMD, FDA} × {0.1, 0.5, 1}; checkpoint = Source-Risk pick per trajectory; single "
             "model = the base run whose pick has the highest Source-Risk. Every target had been seen when the "
             "rule was written, so this is not an independent confirmation.", "",
             "ClinicDB mean deployed Dice by (method, λ): "
             + "; ".join(f"{LABELS[a]} λ{l:g} {v:.3f}" for (a, l), v in sorted(scores.items(), key=lambda x: -x[1])), "",
             f"**Rule picks {LABELS[method]}, λ={lam:g}.**", "",
             "| Target | λ used | Single model (run, step) | Dice | Small Dice | Normal FPR | Empty FPR | 6-run mean Dice [min, max] | Gain vs SourceOnly, 6 runs [95% CI] | SourceOnly single model Dice / FPR |",
             "|---|---:|---|---:|---:|---:|---:|---|---|---|"]
    for s in TARGETS:
        t = targets[s]
        used = lam if t.keys(method, lam) else 0.5
        keys = t.keys(method, used)
        picks = [t.picks[k] for k in keys]
        single = max(picks, key=lambda r: r["validators"]["source_risk"])
        source = max((t.picks[k] for k in t.keys("source_only")), key=lambda r: r["validators"]["source_risk"])
        dice = [t.measure(p, "dice").mean() for p in picks]
        fpr = lambda r: f"{100*t.measure(r, 'fpr_any').mean():.1f}%" if t.has("fpr_any") else "–"
        empty = f"{100*t.measure(single, 'empty').mean():.1f}%" if t.has("empty") else "–"
        g = t.gain(method, "source_only", "dice", lam=used)
        lines.append(f"| {NAMES[s]} | {used:g}" + (" (only λ run)" if used != lam else "")
                     + f" | {base_run(single)} step {single['step']} | {t.measure(single, 'dice').mean():.3f} | "
                     f"{t.measure(single, 'small').mean():.3f} | {fpr(single)} | {empty} | "
                     f"{np.mean(dice):.3f} [{min(dice):.3f}, {max(dice):.3f}] | {interval(g)} ({g[3]}/{g[4]}) | "
                     f"{t.measure(source, 'dice').mean():.3f} / {fpr(source)} |")
    return lines + [""]


def cells(t, algorithm, reference, lam=None):
    """Paired Dice, small-polyp Dice and (where normals exist) normal FPR in points."""
    out = [interval(t.gain(algorithm, reference, "dice", lam)), interval(t.gain(algorithm, reference, "small", lam))]
    if t.has("fpr_any"):
        out.append(interval(t.gain(algorithm, reference, "fpr_any", lam), 100, 1))
    return out


def controls_table(targets):
    lines = ["# Phase 2: source-only continued training (amendment v2)", "",
             "Source-only continuation = the control's source batches without the target forward, so target "
             "images never reach the network (one set of 6 runs, scored on every target). Paired differences "
             "(same base run; lambdas averaged) with 95% intervals; normal FPR in percentage points.", ""]
    rows = [("continue", "continue_source", "Control − source-only continuation (target images in BatchNorm)"),
            ("continue_source", "source_only", "Source-only continuation − SourceOnly (more source training)"),
            ("dacs", "continue_source", "DACS − source-only continuation"),
            ("minent_mmd", "continue_source", "MinEnt+MMD − source-only continuation"),
            ("fda", "continue_source", "FDA − source-only continuation")]
    for s in TARGETS:
        t = targets[s]
        if not t.keys("continue_source"):
            continue
        normal = t.has("fpr_any")
        level = (f"Dice {t.level('continue_source', 'dice'):.3f}, small {t.level('continue_source', 'small'):.3f}"
                 + (f", normal FPR {100*t.level('continue_source', 'fpr_any'):.1f}%" if normal else ""))
        lines += [f"## {NAMES[s]} (source-only continuation: {level})", "",
                  "| Comparison | Dice | Small-polyp Dice" + (" | Normal FPR (points)" if normal else "") + " |",
                  "|---|---|---" + ("|---" if normal else "") + "|"]
        for a, ref, label in rows:
            if t.keys(a):
                lines.append(f"| {label} | " + " | ".join(cells(t, a, ref)) + " |")
        lines.append("")
    return lines


def ablation_table(targets):
    lines = ["# Phase 2: DACS ablations (amendment v2)", "",
             "λ = 0.5 only; checkpoint spacing as the existing DACS runs (ClinicDB, ETIS, ColonDB 200; PolypGen, FAR 400 steps). "
             "Without paste: teacher pseudo-label loss on jittered target images. Without confidence weight: "
             "weight 1 outside the pasted polyp. Paired differences DACS − variant, same base run; 95% intervals.", ""]
    for s in ABLATION_TARGETS:
        t = targets[s]
        normal = t.has("fpr_any")
        lines += [f"## {NAMES[s]}", "",
                  "| Model (λ 0.5) | Dice | Small-polyp Dice" + (" | Normal FPR" if normal else "") + " |",
                  "|---|---:|---:" + ("|---:" if normal else "") + "|"]
        for a in ("dacs", *ABLATIONS):
            if t.keys(a, 0.5):
                lines.append(f"| {LABELS[a]} | {t.level(a, 'dice', 0.5):.3f} | {t.level(a, 'small', 0.5):.3f}"
                             + (f" | {100*t.level(a, 'fpr_any', 0.5):.1f}%" if normal else "") + " |")
        lines += ["", "| Comparison | Dice | Small-polyp Dice" + (" | Normal FPR (points)" if normal else "") + " |",
                  "|---|---|---" + ("|---" if normal else "") + "|"]
        for a in ABLATIONS:
            if t.keys(a, 0.5):
                lines.append(f"| DACS − {LABELS[a]} | " + " | ".join(cells(t, "dacs", a, 0.5)) + " |")
        lines.append("")
    return lines


def main():
    table, units = load_pools(POOLS, features=False)
    targets = {}
    for s in TARGETS:
        targets[s] = Target(s, table[s], units[s])
    paired_tables = {s: paired(table[s], units[s]) for s in TARGETS}
    (RESULTS/"h2.md").write_text("\n".join(h2_table(targets)), encoding="utf-8")
    (RESULTS/"endpoints.md").write_text("\n".join(endpoint_table(targets, paired_tables)), encoding="utf-8")
    (RESULTS/"deployment.md").write_text("\n".join(deployment(targets)), encoding="utf-8")
    if any(targets[s].keys("continue_source") for s in TARGETS):
        (RESULTS/"controls.md").write_text("\n".join(controls_table(targets)), encoding="utf-8")
    if any(targets[s].keys(a) for s in ABLATION_TARGETS for a in ABLATIONS):
        (RESULTS/"ablations.md").write_text("\n".join(ablation_table(targets)), encoding="utf-8")
    print("wrote", RESULTS/"h2.md", RESULTS/"endpoints.md", RESULTS/"deployment.md")


if __name__ == "__main__":
    main()
