"""Exploratory native-mask comparison of one variant and frozen A1 baselines."""
import argparse
import json
from pathlib import Path

import yaml

from datasets.segmentation import SegmentationDataset
from scripts.check_protocol import sha256
from scripts.evaluate_multiscale import checkpoint_model, evaluate, summarize
from utils import choose_device


def compare(default_protocol, default_run_root, default_output, augmentation,
            loss_name, method_name):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=default_protocol)
    parser.add_argument("--original-selection", default="runs/real/followup/guarded_selection.json")
    parser.add_argument("--run-root", default=default_run_root)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    protocol_hash = sha256(args.protocol)
    if sha256("protocol.yaml") != protocol["original_protocol_sha256"]:
        raise ValueError("Original experiment protocol changed")
    if sha256(args.original_selection) != protocol["original_source_selection_sha256"]:
        raise ValueError("Original source checkpoint selection changed")
    selection = json.loads(Path(args.original_selection).read_text(encoding="utf-8"))
    original = {run["seed"]: run["methods"]["source"] for run in selection["runs"]
                if run["augmentation"] == "a1"}
    if sorted(original) != sorted(protocol["seeds"]):
        raise ValueError("Original baseline seeds differ")
    run_root = Path(args.run_root)
    candidates = {}
    for seed in protocol["seeds"]:
        run = run_root / f"{augmentation}_seed{seed}"
        metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["protocol_sha256"] != protocol_hash:
            raise ValueError(f"Training protocol differs for seed {seed}")
        path = run / "best_source.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        candidates[seed] = {"path": str(path), "sha256": sha256(path)}
    domains = {
        "CVC-ClinicDB": SegmentationDataset("data/polyp_png", "target_test", 256, 3).pairs,
        "CVC-ColonDB": SegmentationDataset("data/TestDataset/TestDataset", "CVC-ColonDB", 256, 3).pairs,
        "ETIS-LaribPolypDB": SegmentationDataset("data/TestDataset/TestDataset", "ETIS-LaribPolypDB", 256, 3).pairs,
    }
    expected_counts = {"CVC-ClinicDB": 427, "CVC-ColonDB": 380, "ETIS-LaribPolypDB": 196}
    if {name: len(pairs) for name, pairs in domains.items()} != expected_counts:
        raise ValueError("Evaluation dataset counts changed")
    device = choose_device(args.device)
    results = []
    for seed in protocol["seeds"]:
        baseline = original[seed]
        source_model, source_size = checkpoint_model(
            baseline["checkpoint"], baseline["checkpoint_sha256"], seed, "a1", device)
        new = candidates[seed]
        new_model, new_size = checkpoint_model(
            new["path"], new["sha256"], seed, augmentation, device, loss_name)
        if source_size != new_size or new_size != int(protocol["training_image_size"]):
            raise ValueError("Compared different input sizes")
        for name, pairs in domains.items():
            source_rows = evaluate(source_model, source_size, pairs,
                                   float(protocol["prediction_threshold"]), device)
            new_rows = evaluate(new_model, new_size, pairs,
                                float(protocol["prediction_threshold"]), device)
            if [row["sample_id"] for row in source_rows] != [row["sample_id"] for row in new_rows]:
                raise ValueError("Compared different image pairs")
            cutoff = float(protocol["small_polyp_area_cutoff"])
            source_summary = summarize(source_rows, cutoff)
            new_summary = summarize(new_rows, cutoff)
            row = {"dataset": name, "seed": seed, "source": source_summary,
                   method_name: new_summary,
                   "dice_difference": new_summary["mean_dice"] - source_summary["mean_dice"],
                   "small_dice_difference": (new_summary["small_mean_dice"]
                                             - source_summary["small_mean_dice"]),
                   "source_per_image": source_rows, f"{method_name}_per_image": new_rows}
            results.append(row)
            print(name, seed, "source", f"{source_summary['mean_dice']:.4f}",
                  method_name, f"{new_summary['mean_dice']:.4f}",
                  "small delta", f"{row['small_dice_difference']:+.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    result = {"status": protocol["evaluation_status"],
              "protocol_sha256": protocol_hash,
              "original_selection_sha256": sha256(args.original_selection),
              f"{method_name}_checkpoints": candidates, "results": results}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    compare("tversky_protocol.yaml", "runs/real/tversky",
            "runs/real/tversky/evaluation.json", "a1", "bce_tversky", "tversky")
