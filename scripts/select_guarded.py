"""Exploratory follow-up selection using source quality and empty-mask guards.

Only the frozen source-validation and unlabeled target-selection candidate table is
read here. The rule was designed after the original CVC target-test analysis, so
its result on that test set is exploratory; external data are evaluated separately.
"""
import argparse
import json
from pathlib import Path

import yaml

from scripts.check_protocol import sha256


def choose_guarded(rows, source_dice_tolerance, empty_fraction_margin):
    if not rows or not 0 <= source_dice_tolerance <= 1 or not 0 <= empty_fraction_margin <= 1:
        raise ValueError("Invalid candidates or guard parameters")
    source = min(rows, key=lambda row: (-row["source_val_dice"], row["checkpoint"]))
    eligible = [row for row in rows
                if row["source_val_dice"] >= source["source_val_dice"] - source_dice_tolerance
                and row["empty_prediction_fraction"] <=
                source["empty_prediction_fraction"] + empty_fraction_margin]
    guarded = min(eligible, key=lambda row: (
        row["target_instability"], -row["source_val_dice"], row["checkpoint"]
    ))
    return source, guarded, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="followup_protocol.yaml")
    parser.add_argument("--selection-dir", default="runs/real/selections")
    parser.add_argument("--output", default="runs/real/followup/guarded_selection.json")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if sha256("protocol.yaml") != protocol["original_protocol_sha256"]:
        raise ValueError("Original six-run protocol differs from the follow-up lock")
    results = []
    for name, expected_hash in protocol["original_selection_sha256"].items():
        path = Path(args.selection_dir) / f"{name}.json"
        if sha256(path) != expected_hash:
            raise ValueError(f"Original selection differs from the follow-up lock: {name}")
        selection = json.loads(path.read_text(encoding="utf-8"))
        rows = selection["candidates"]
        if len(rows) != 30 or sorted(row["epoch"] for row in rows) != list(range(1, 31)):
            raise ValueError(f"Incomplete candidate pool: {name}")
        if f"{selection['augmentation']}_seed{selection['seed']}" != name:
            raise ValueError(f"Candidate identity differs: {name}")
        source, guarded, eligible = choose_guarded(
            rows, float(protocol["source_dice_tolerance"]),
            float(protocol["empty_fraction_margin"]),
        )
        if source["checkpoint"] != selection["methods"]["source"]:
            raise ValueError(f"Source comparator differs from original selection: {name}")
        original_stability = next(row for row in rows
                                  if row["checkpoint"] == selection["methods"]["stability"])
        methods = {}
        for method, row in (("source", source), ("original_stability", original_stability),
                            ("guarded", guarded)):
            checkpoint = Path(row["checkpoint"])
            if not checkpoint.is_file() or checkpoint.resolve().parent != (
                Path(selection["run_dir"]) / "checkpoints").resolve():
                raise ValueError(f"Checkpoint outside the frozen candidate pool: {checkpoint}")
            methods[method] = {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256(checkpoint),
                "epoch": row["epoch"],
                "source_val_dice": row["source_val_dice"],
                "target_instability": row["target_instability"],
                "empty_prediction_fraction": row["empty_prediction_fraction"],
                "mean_foreground_fraction": row["mean_foreground_fraction"],
            }
        results.append({"run": name, "augmentation": selection["augmentation"],
                        "seed": selection["seed"], "eligible_count": len(eligible),
                        "methods": methods})
    result = {
        "status": "exploratory_followup_after_original_cvc_test",
        "source_dice_tolerance": protocol["source_dice_tolerance"],
        "empty_fraction_margin": protocol["empty_fraction_margin"],
        "runs": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in results:
        print(row["run"], "source", row["methods"]["source"]["epoch"],
              "guarded", row["methods"]["guarded"]["epoch"],
              "eligible", row["eligible_count"])
    print("guarded_selection_sha256:", sha256(output))


if __name__ == "__main__":
    main()
