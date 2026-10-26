"""Describe candidate rankings and failure patterns after final target evaluation.

These diagnostics use target-test labels and must never feed back into selection.
"""
import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.check_protocol import sha256


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman(left, right):
    """Pearson correlation of average ranks, including tied observations."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Ranked series must have equal lengths of at least two")
    x, y = average_ranks(left), average_ranks(right)
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    x_var = sum((value - x_mean) ** 2 for value in x)
    y_var = sum((value - y_mean) ** 2 for value in y)
    if x_var == 0 or y_var == 0:
        return None
    return sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / math.sqrt(x_var * y_var)


def candidate_target_dice(candidate, cell_dir, selected_reports):
    checkpoint = str(Path(candidate["checkpoint"]).resolve())
    path = cell_dir / f"epoch_{candidate['epoch']:03d}.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else next(
        (report for report in selected_reports if report["checkpoint"] == checkpoint), None
    )
    if report is None or report["checkpoint"] != checkpoint or report["n_images"] != 427:
        raise ValueError(f"Missing or mismatched candidate target report: {checkpoint}")
    return report["mean_dice"]


def diagnose_cell(selection, cell_dir):
    reports = {method: json.loads((cell_dir / f"{method}.json").read_text(encoding="utf-8"))
               for method in ("source", "stability", "confidence")}
    candidates = selection["candidates"]
    if len(candidates) != 30:
        raise ValueError(f"Expected 30 candidates in {cell_dir}")
    actual = [candidate_target_dice(row, cell_dir, reports.values()) for row in candidates]
    source = {row["sample_id"]: row for row in reports["source"]["per_image"]}
    stability = {row["sample_id"]: row for row in reports["stability"]["per_image"]}
    if source.keys() != stability.keys() or len(source) != 427:
        raise ValueError(f"Target-test image IDs differ in {cell_dir}")
    groups = defaultdict(list)
    for sample_id, row in stability.items():
        baseline = source[sample_id]
        if row["group_id"] != baseline["group_id"] or not row["group_id"]:
            raise ValueError(f"Target sequence ID differs in {cell_dir}: {sample_id}")
        groups[row["group_id"]].append(row["dice"] - baseline["dice"])
    group_means = {group: statistics.mean(values) for group, values in groups.items()}
    worst_group = min(group_means, key=lambda group: (group_means[group], group))
    return {
        "run": cell_dir.name,
        "candidate_count": len(candidates),
        "target_image_count": len(source),
        "target_group_count": len(groups),
        "spearman_source_dice_vs_target_dice_posthoc": spearman(
            [row["source_val_dice"] for row in candidates], actual),
        "spearman_negative_instability_vs_target_dice_posthoc": spearman(
            [-row["target_instability"] for row in candidates], actual),
        "spearman_negative_entropy_vs_target_dice_posthoc": spearman(
            [-row["mean_entropy"] for row in candidates], actual),
        "stability_zero_dice_image_count": sum(row["dice"] == 0 for row in stability.values()),
        "stability_empty_prediction_count": sum(row["foreground_fraction"] == 0
                                                 for row in stability.values()),
        "worst_group_id": worst_group,
        "worst_group_image_count": len(groups[worst_group]),
        "worst_group_mean_paired_dice_difference": group_means[worst_group],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", default="runs/real/selections")
    parser.add_argument("--final-dir", default="runs/real/final")
    parser.add_argument("--output", default="runs/real/final/posthoc_diagnostics.json")
    args = parser.parse_args()
    final_dir = Path(args.final_dir)
    lock = json.loads((final_dir / "final_lock.json").read_text(encoding="utf-8"))
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    results = []
    for row in summary["runs"]:
        name = row["run"]
        selection_path = Path(args.selection_dir) / f"{name}.json"
        if sha256(selection_path) != lock["selection_sha256"][name]:
            raise ValueError(f"Selection differs from frozen final analysis: {name}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        results.append(diagnose_cell(selection, final_dir / name))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"runs": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(results)} post-hoc diagnostic rows to {output}")


if __name__ == "__main__":
    main()
