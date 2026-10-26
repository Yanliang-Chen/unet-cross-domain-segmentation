"""Exploratory native-mask comparison of multiscale U-Net and frozen A1 baselines."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from datasets.segmentation import SegmentationDataset, load_image
from models.unet import build_model
from scripts.check_protocol import sha256
from utils import choose_device, per_image_metrics


def evaluate(model, image_size, pairs, threshold, device):
    rows = []
    model.eval()
    with torch.inference_mode():
        for image_path, mask_path in pairs:
            image = load_image(image_path, image_size, 3)[None].to(device)
            with Image.open(mask_path) as mask:
                target = torch.from_numpy((np.asarray(mask.convert("L")) > 127).copy())
            target = target[None, None].to(device=device, dtype=torch.float32)
            logits = model(image)
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear",
                                   align_corners=False)
            metrics = per_image_metrics(logits, target, threshold)[0]
            rows.append({"sample_id": image_path.stem,
                         "ground_truth_fraction": float(target.mean()), **metrics})
    return rows


def checkpoint_model(path, expected_hash, expected_seed, expected_augmentation, device,
                     expected_loss="bce_dice"):
    if sha256(path) != expected_hash:
        raise ValueError(f"Checkpoint differs from the frozen input: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state["config"]
    if (int(cfg["seed"]) != expected_seed or cfg["augmentation"] != expected_augmentation
            or cfg.get("loss", "bce_dice") != expected_loss):
        raise ValueError(f"Checkpoint identity differs: {path}")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model, int(cfg["image_size"])


def summarize(rows, small_cutoff):
    keys = ("dice", "iou", "precision", "recall", "foreground_fraction")
    small = [row for row in rows if row["ground_truth_fraction"] < small_cutoff]
    return {"n_images": len(rows), "n_small": len(small),
            **{f"mean_{key}": statistics.mean(row[key] for row in rows) for key in keys},
            "small_mean_dice": statistics.mean(row["dice"] for row in small),
            "small_mean_recall": statistics.mean(row["recall"] for row in small),
            "empty_prediction_fraction": statistics.mean(
                row["foreground_fraction"] == 0 for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="multiscale_protocol.yaml")
    parser.add_argument("--original-selection", default="runs/real/followup/guarded_selection.json")
    parser.add_argument("--run-root", default="runs/real/multiscale")
    parser.add_argument("--output", default="runs/real/multiscale/evaluation.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
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
    protocol_hash = sha256(args.protocol)
    for seed in protocol["seeds"]:
        run = run_root / f"a2_seed{seed}"
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
        new_model, new_size = checkpoint_model(new["path"], new["sha256"], seed, "a2", device)
        if new_size != int(protocol["training_image_size"]):
            raise ValueError("Multiscale checkpoint image size changed")
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
                   "multiscale": new_summary,
                   "dice_difference": new_summary["mean_dice"] - source_summary["mean_dice"],
                   "small_dice_difference": (new_summary["small_mean_dice"]
                                             - source_summary["small_mean_dice"]),
                   "source_per_image": source_rows, "multiscale_per_image": new_rows}
            results.append(row)
            print(name, seed, "source", f"{source_summary['mean_dice']:.4f}",
                  "multiscale", f"{new_summary['mean_dice']:.4f}",
                  "small delta", f"{row['small_dice_difference']:+.4f}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    result = {"status": protocol["evaluation_status"],
              "protocol_sha256": protocol_hash,
              "original_selection_sha256": sha256(args.original_selection),
              "multiscale_checkpoints": candidates, "results": results}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Saved", output)


if __name__ == "__main__":
    main()
