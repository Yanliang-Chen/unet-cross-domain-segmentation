"""Evaluate a frozen six-run matrix, then report paired and post-hoc results."""
import argparse
import copy
import csv
import json
import random
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from scripts.check_protocol import check_protocol, sha256
from select_model import select_rows


def paired_dice(source, selected):
    left = {row["sample_id"]: row for row in source["per_image"]}
    right = {row["sample_id"]: row for row in selected["per_image"]}
    if left.keys() != right.keys():
        raise ValueError("The two methods evaluated different target images")
    if any(left[name]["group_id"] != right[name]["group_id"] for name in left):
        raise ValueError("Sequence IDs differ between method reports")
    return [{"sample_id": name, "group_id": left[name]["group_id"],
             "dice_difference": right[name]["dice"] - left[name]["dice"]}
            for name in sorted(left)]


def cluster_interval(rows, repeats=2000, seed=20261026):
    """Percentile interval resampling videos, preserving all frames in a video."""
    groups = defaultdict(list)
    for row in rows:
        if not row["group_id"]:
            raise ValueError("A target image has no sequence ID")
        groups[row["group_id"]].append(row["dice_difference"])
    ids = sorted(groups)
    if len(ids) < 2:
        raise ValueError("At least two target sequences are needed")
    rng = random.Random(seed)
    estimates = []
    for _ in range(repeats):
        sampled = [groups[rng.choice(ids)] for _ in ids]
        estimates.append(sum(map(sum, sampled)) / sum(map(len, sampled)))
    estimates.sort()
    return [estimates[int(0.025 * (repeats - 1))],
            estimates[int(0.975 * (repeats - 1))]]


def run_evaluation(checkpoint, selection, method, output, data_dir, protocol, device,
                   manifest_sha256):
    expected_checkpoint = (json.loads(Path(selection).read_text(encoding="utf-8"))[
        "methods"][method] if selection else checkpoint)
    expected_checkpoint = str(Path(expected_checkpoint).resolve())
    if output.exists() and output.with_suffix(".csv").exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (existing["checkpoint"] != expected_checkpoint or
                existing["split"] != "target_test" or
                existing["split_manifest_sha256"] != manifest_sha256):
            raise ValueError(f"Existing evaluation does not match the frozen inputs: {output}")
        return existing
    command = [sys.executable, "evaluate.py", "--data-dir", str(data_dir),
               "--protocol", str(protocol), "--device", device,
               "--final-evaluation", "--output", str(output),
               "--prevalidated-manifest-sha256", manifest_sha256]
    if selection:
        command += ["--selection", str(selection), "--method", method]
    else:
        command += ["--checkpoint", str(checkpoint)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"Evaluation failed for {expected_checkpoint}:\n"
                           f"{completed.stderr[-4000:]}\n{completed.stdout[-2000:]}")
    return json.loads(output.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="protocol.yaml")
    parser.add_argument("--data-dir", default="data/polyp_png")
    parser.add_argument("--selection-dir", default="runs/real/selections")
    parser.add_argument("--output-dir", default="runs/real/final")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    manifest_info = check_protocol(args.protocol, args.data_dir)
    protocol_digest = sha256(args.protocol)
    selections = {}
    for augmentation in protocol["augmentations"]:
        for seed in protocol["seeds"]:
            name = f"{augmentation}_seed{seed}"
            path = Path(args.selection_dir) / f"{name}.json"
            selection = json.loads(path.read_text(encoding="utf-8"))
            if (selection["augmentation"], selection["seed"]) != (augmentation, seed):
                raise ValueError(f"Selection identity mismatch: {path}")
            if float(selection["tolerance"]) != float(protocol["source_dice_tolerance"]):
                raise ValueError(f"Selection tolerance mismatch: {path}")
            if Path(selection["target_images"]).resolve() != (
                    Path(args.data_dir) / "target_select" / "images").resolve():
                raise ValueError(f"Selection target path mismatch: {path}")
            candidates = selection["candidates"]
            if len(candidates) != protocol["epochs"] or sorted(
                    row["epoch"] for row in candidates) != list(range(1, protocol["epochs"] + 1)):
                raise ValueError(f"Candidate pool is incomplete: {path}")
            if any(Path(row["checkpoint"]).resolve().parent !=
                   (Path(selection["run_dir"]) / "checkpoints").resolve()
                   for row in candidates):
                raise ValueError(f"Candidate outside the declared run: {path}")
            recomputed, _ = select_rows(copy.deepcopy(candidates),
                                        float(protocol["source_dice_tolerance"]))
            if selection["methods"] != {key: row["checkpoint"]
                                        for key, row in recomputed.items()}:
                raise ValueError(f"Selection methods do not match the frozen rule: {path}")
            metadata = json.loads((Path(selection["run_dir"]) / "run_metadata.json").read_text(
                encoding="utf-8"))
            if metadata["protocol_sha256"] != protocol_digest or (
                    metadata["split_manifest_sha256"] != manifest_info["manifest_sha256"]):
                raise ValueError(f"Run was trained under a different protocol/data split: {path}")
            selections[name] = (path, selection)
    output_dir = Path(args.output_dir)
    lock = {"protocol_sha256": protocol_digest,
            "split_manifest_sha256": manifest_info["manifest_sha256"],
            "selection_sha256": {name: sha256(path) for name, (path, _) in selections.items()}}
    lock_path = output_dir / "final_lock.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not lock_path.exists() or json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise FileExistsError(f"Final output is not empty with the same frozen inputs: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
        return
    summary = []
    for name, (path, selection) in selections.items():
        print(f"Evaluating {name}", flush=True)
        cell_dir = output_dir / name
        cell_dir.mkdir(exist_ok=True)
        reports = {}
        by_checkpoint = {}
        for method in ("source", "stability", "confidence"):
            report = run_evaluation(None, path, method, cell_dir / f"{method}.json",
                                    args.data_dir, args.protocol, args.device,
                                    manifest_info["manifest_sha256"])
            reports[method] = report
            by_checkpoint[report["checkpoint"]] = report
        oracle = None
        for candidate in selection["candidates"]:
            checkpoint = str(Path(candidate["checkpoint"]).resolve())
            if checkpoint not in by_checkpoint:
                report = run_evaluation(checkpoint, None, None,
                                        cell_dir / f"epoch_{candidate['epoch']:03d}.json",
                                        args.data_dir, args.protocol, args.device,
                                        manifest_info["manifest_sha256"])
                by_checkpoint[checkpoint] = report
            report = by_checkpoint[checkpoint]
            if oracle is None or (report["mean_dice"], checkpoint) > (
                    oracle["mean_dice"], oracle["checkpoint"]):
                oracle = report
        differences = paired_dice(reports["source"], reports["stability"])
        with (cell_dir / "paired_dice.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=differences[0].keys())
            writer.writeheader()
            writer.writerows(differences)
        summary.append({
            "run": name, "seed": selection["seed"], "augmentation": selection["augmentation"],
            "source_dice": reports["source"]["mean_dice"],
            "stability_dice": reports["stability"]["mean_dice"],
            "confidence_dice": reports["confidence"]["mean_dice"],
            "oracle_dice_posthoc": oracle["mean_dice"],
            "oracle_checkpoint_posthoc": oracle["checkpoint"],
            "paired_dice_difference": statistics.mean(
                row["dice_difference"] for row in differences),
            "sequence_bootstrap_95_ci": cluster_interval(
                differences, repeats=int(protocol["bootstrap_repeats"]),
                seed=int(protocol["bootstrap_seed"])),
        })
    aggregate = {}
    for augmentation in protocol["augmentations"]:
        rows = [row for row in summary if row["augmentation"] == augmentation]
        values = [row["paired_dice_difference"] for row in rows]
        aggregate[augmentation] = {
            "mean_paired_dice_difference_across_seeds": statistics.mean(values),
            "sd_across_seeds": statistics.stdev(values),
            "n_seeds": len(values),
        }
    result = {"runs": summary, "aggregate": aggregate,
              "oracle_warning": "Oracle uses target labels after protocol freeze; it is only a post-hoc upper bound."}
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
