"""Selection gap and validator reliability over one or more scored candidate pools.

Usage:
  python -m benchmark.analyze --pools source_only [--name source_only] [--write-ensembles]
  python -m benchmark.analyze --pools source_only uda_adabn uda_minent --name clinicdb_all
Writes runs/benchmark/analysis/<name>.json and .md. With --write-ensembles it also
registers, per scenario, the equal-weight ensemble of the across-pool validator picks
as pool ensemble_<name>; score that pool, then rerun this script to include it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, spearmanr

from benchmark.score import REGISTRY, SCORES
from benchmark.validators import auroc, feature_validators, validator_scores

OUT = Path("runs/benchmark/analysis")
TARGETS = ("clinicdb", "colondb", "etis", "polypgen", "far")
VALIDATORS = ("source_risk", "iwcv", "dev", "devn", "entropy", "infomax", "bnm", "mcc", "snd",
              "mixval", "mixval_fg", "cmr_nhd_gauss", "cmr_ei_gauss", "cmr_nhd_fg_gauss",
              "cmr_ei_fg_gauss", "cmr_nhd_ttd", "cmr_ei_ttd", "cmr_nhd_fg_ttd", "cmr_ei_fg_ttd")
GUARD = (0.25, 4.0)
SMALL = 0.02
BOOTSTRAP, BOOTSTRAP_SEED = 2000, 20261031
BUDGETS, BUDGET_REPEATS, BUDGET_SEED = (0.01, 0.02, 0.03, 0.04, 0.05), 10, 20261101


def candidate_metrics(scores, meta):
    test, kind = scores["is_test"], meta["kind"]
    polyp, normal = test & (kind == "polyp"), test & (kind == "normal")
    small = polyp & (meta["gt_fraction"] < SMALL)
    groups = meta["group"][polyp]
    out = {"dice": float(np.mean(scores["dice"][polyp])),
           "iou": float(np.mean(scores["iou"][polyp])),
           "small_dice": float(np.mean(scores["dice"][small])) if small.any() else None,
           "n_polyp": int(polyp.sum()), "n_small": int(small.sum()), "n_normal": int(normal.sum()),
           "_dice": scores["dice"][polyp].astype(np.float32)}
    if groups.size and all(groups):
        out["group_dice"] = float(np.mean([np.mean(scores["dice"][polyp][groups == g])
                                           for g in np.unique(groups)]))
    if normal.any():
        out["fpr_any"] = float(np.mean(scores["fg_native"][normal] > 0))
        out["fpr_area"] = float(np.mean(scores["fg_native"][normal] >= 0.001))
        out["auroc"] = auroc(scores["max_prob"][polyp], scores["max_prob"][normal])
    validation = scores["is_validation"]
    out["validators"] = validator_scores({k: scores[k][validation] for k in scores.files})
    return out


def load_pools(pools, features=True):
    table, units = {}, {}
    for pool in pools:
        candidates = json.loads((REGISTRY/f"{pool}.json").read_text(encoding="utf-8"))
        source, source_extra = {}, {}
        for c in candidates:
            path = SCORES/pool/"source_val"/f"{c['id']}.npz"
            if path.exists():
                dice = np.load(path)["dice"]
                source[c["id"]] = float(np.nanmean(dice))
                extra = path.with_suffix(".extra.npz")
                if extra.exists():
                    source_extra[c["id"]] = (np.load(extra)["z"], dice)
        for scenario in TARGETS:
            meta_path = SCORES/f"meta_{scenario}.npz"
            if not meta_path.exists():
                continue
            meta = np.load(meta_path)
            polyp = meta["kind"] == "polyp"
            units[scenario] = meta["group"][polyp]
            for c in candidates:
                path = SCORES/pool/scenario/f"{c['id']}.npz"
                if not path.exists():
                    continue
                scores = np.load(path)
                if not np.array_equal(scores["is_test"][polyp], np.ones(polyp.sum(), bool)):
                    raise ValueError("Polyp images must all be test images")
                row = {**c, "pool": pool, **candidate_metrics(scores, meta)}
                row["validators"]["source_risk"] = source.get(c["id"], float("nan"))
                extra = path.with_suffix(".extra.npz")
                if features and extra.exists() and c["id"] in source_extra:
                    target = np.load(extra)
                    z_source, source_dice = source_extra[c["id"]]
                    row["validators"].update(feature_validators(z_source, source_dice, target["z"],
                                                                target, target))
                table.setdefault(scenario, []).append(row)
    return table, units


def _correlation(scores, values):
    scores, values = np.asarray(scores, float), np.asarray(values, float)
    keep = np.isfinite(scores) & np.isfinite(values)
    if keep.sum() < 3 or np.ptp(scores[keep]) == 0:
        return None, None
    return float(spearmanr(scores[keep], values[keep])[0]), float(kendalltau(scores[keep], values[keep])[0])


def _unit_means(dice, groups):
    """Per-unit mean Dice: video/patient group where every image has one, else image."""
    if groups.size and all(groups):
        labels, inverse = np.unique(groups, return_inverse=True)
        sums = np.bincount(inverse, weights=dice, minlength=len(labels))
        return sums/np.bincount(inverse, minlength=len(labels))
    return dice


def bootstrap_gap(oracle_dice, pick_dice, groups, rng):
    """95% percentile interval of mean(oracle - pick), resampling groups (or images)."""
    diff = _unit_means(oracle_dice - pick_dice, groups)
    draws = rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))
    means = diff[draws].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def label_budget(rows, rng):
    """Xiong et al.: pick by Dice on a random labelled subset; report on the remaining images."""
    dice = np.stack([r["_dice"] for r in rows])
    n, out = dice.shape[1], {}
    for fraction in BUDGETS:
        k = max(1, round(fraction*n))
        gaps, picks = [], []
        for _ in range(BUDGET_REPEATS):
            labelled = np.zeros(n, bool)
            labelled[rng.choice(n, k, replace=False)] = True
            pick = int(np.argmax(dice[:, labelled].mean(1)))
            rest = dice[:, ~labelled].mean(1)
            gaps.append(float(rest.max()-rest[pick]))
            picks.append(float(rest[pick]))
        out[f"{fraction:g}"] = {"labelled_images": k, "mean_gap": float(np.mean(gaps)),
                                "mean_dice": float(np.mean(picks))}
    return out


def evaluate_pool(rows, groups, guard_reference, rng, intervals):
    dice = np.array([r["dice"] for r in rows])
    oracle = int(np.argmax(dice))
    result = {"n": len(rows), "oracle": _summary(rows[oracle]), "pool_mean_dice": float(dice.mean()),
              "last": _summary(max(rows, key=lambda r: r["step"])), "validators": {}}
    low, high = GUARD[0]*guard_reference, GUARD[1]*guard_reference
    guarded = [i for i, r in enumerate(rows)
               if low <= r["validators"]["foreground_fraction"] <= high]
    for name in VALIDATORS:
        values = np.array([r["validators"].get(name, np.nan) for r in rows], float)
        if not np.isfinite(values).all():
            continue                                  # validator not scored for every candidate
        values = np.where(np.isfinite(values), values, -np.inf)
        pick = int(np.argmax(values))
        rho, tau = _correlation(values, dice)
        entry = {"pick": _summary(rows[pick]), "gap": float(dice[oracle]-dice[pick]),
                 "spearman": rho, "kendall": tau}
        if intervals:
            entry["gap_ci95"] = bootstrap_gap(rows[oracle]["_dice"], rows[pick]["_dice"], groups, rng)
        if "fpr_any" in rows[0]:
            entry["spearman_vs_negative_fpr"] = _correlation(values, [-r["fpr_any"] for r in rows])[0]
        if guarded:
            g = max(guarded, key=lambda i: values[i])
            entry["guarded_pick"] = _summary(rows[g])
            entry["guarded_gap"] = float(dice[oracle]-dice[g])
        entry["guard_kept"] = len(guarded)
        result["validators"][name] = entry
    if intervals:
        result["label_budget"] = label_budget(rows, rng)
    return result


def _summary(row):
    keys = ("id", "algorithm", "dice", "group_dice", "small_dice", "fpr_any", "fpr_area", "auroc")
    return {k: row[k] for k in keys if k in row}


def analyze(pools, name, write_ensembles):
    table, units = load_pools(pools)
    source_meta = np.load(SCORES/"meta_source_val.npz")
    guard_reference = float(np.mean(source_meta["gt_fraction"]))
    report = {"pools": pools, "guard_reference_mask_fraction": guard_reference, "scenarios": {}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ensembles = []
    for scenario, rows in table.items():
        subsets = {"across_all": rows}
        for key in ("algorithm", "augmentation", "run"):
            values = sorted({r[key] for r in rows})
            if len(values) > 1 or key == "run":
                for value in values:
                    subsets[f"{key}={value}"] = [r for r in rows if r[key] == value]
        report["scenarios"][scenario] = {
            label: evaluate_pool(subset, units[scenario], guard_reference, rng,
                                 intervals=not label.startswith("run="))
            for label, subset in subsets.items()}
        whole = report["scenarios"][scenario]["across_all"]
        picks = sorted({e["pick"]["id"] for e in whole["validators"].values()})
        by_id = {r["id"]: r for r in rows}
        ensembles.append({"id": f"ensemble_{scenario}", "algorithm": f"ensemble_{name}", "lam": 0.0,
                          "augmentation": "mixed", "seed": 0, "step": 0, "run": "ensemble",
                          "members": picks, "scenarios": [scenario, "source_val"],
                          "checkpoints": [p for i in picks for p in by_id[i]["checkpoints"]]})
    ens_pool = f"ensemble_{name}"
    if (REGISTRY/f"{ens_pool}.json").exists():
        ens_table, _ = load_pools([ens_pool])
        for scenario, rows in ens_table.items():
            if scenario in report["scenarios"]:
                report["scenarios"][scenario]["validator_ensemble"] = _summary(rows[0])
    if write_ensembles:
        (REGISTRY/f"{ens_pool}.json").write_text(json.dumps(ensembles, indent=1)+"\n", encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f"{name}.json").write_text(json.dumps(report, indent=1)+"\n", encoding="utf-8")
    (OUT/f"{name}.md").write_text(markdown(name, report), encoding="utf-8")
    print(markdown(name, report))


def _fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def markdown(name, report):
    lines = [f"# Selection gap: `{name}` (pools: {', '.join(report['pools'])})", ""]
    for scenario, pools in report["scenarios"].items():
        whole = pools["across_all"]
        runs = [v for k, v in pools.items() if k.startswith("run=")]
        ens = pools.get("validator_ensemble")
        lines += [f"## {scenario} (n={whole['n']})", "",
                  f"Oracle Dice {_fmt(whole['oracle']['dice'])} ({whole['oracle']['id']}); "
                  f"pool mean {_fmt(whole['pool_mean_dice'])}"
                  + (f"; validator ensemble Dice {_fmt(ens['dice'])}"
                     + (f", FPR {_fmt(ens.get('fpr_any'))}" if "fpr_any" in ens else "") if ens else ""), "",
                  "| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
        for v, e in whole["validators"].items():
            in_runs = [r["validators"][v] for r in runs if v in r["validators"]]
            run_gap = np.mean([x["gap"] for x in in_runs]) if in_runs else None
            run_rho = [x["spearman"] for x in in_runs if x["spearman"] is not None]
            ci = e.get("gap_ci95")
            lines.append(f"| {v} | {e['pick'].get('algorithm', '')} | {_fmt(e['pick']['dice'])} | {_fmt(e['gap'])}"
                         + (f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "")
                         + f" | {_fmt(e.get('guarded_gap'))} | {_fmt(e['spearman'], 2)} | {_fmt(e['kendall'], 2)} | "
                         f"{_fmt(e['pick'].get('fpr_any'))} | {_fmt(run_gap)} | "
                         f"{_fmt(np.mean(run_rho) if run_rho else None, 2)} |")
        budget = whole.get("label_budget", {})
        if budget:
            lines += ["", "Label budget (pick by Dice on labelled subset, gap on remaining images): "
                      + "; ".join(f"{float(f)*100:.0f}% ({b['labelled_images']} img) gap {_fmt(b['mean_gap'])}"
                                  for f, b in budget.items())]
        algorithms = {k.split("=", 1)[1]: p for k, p in pools.items() if k.startswith("algorithm=")}
        if algorithms:
            lines += ["", "| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |",
                      "|---|---:|---:|---:|---:|---:|"]
            for algorithm, p in algorithms.items():
                lines.append(f"| {algorithm} | {p['n']} | {_fmt(p['oracle']['dice'])} | "
                             + " | ".join(_fmt(p["validators"][v]["pick"]["dice"])
                                          for v in ("source_risk", "entropy", "cmr_ei_gauss")) + " |")
        run_last = np.mean([r["oracle"]["dice"]-r["last"]["dice"] for r in runs]) if runs else None
        lines += ["", f"Last-checkpoint reference: within-run gap {_fmt(run_last)}.", ""]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pools", nargs="+", required=True)
    parser.add_argument("--name")
    parser.add_argument("--write-ensembles", action="store_true")
    args = parser.parse_args()
    analyze(args.pools, args.name or "_".join(args.pools), args.write_ensembles)


if __name__ == "__main__":
    main()
