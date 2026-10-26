"""Evaluate frozen exploratory selections on two previously unused test datasets.

This does not revise the original CVC-ClinicDB protocol or its reported result.
External masks are loaded only after the follow-up protocol and selection hashes match.
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from datasets.files import index_images
from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from scripts.check_protocol import sha256
from utils import choose_device, per_image_metrics


def audit_external(protocol):
    archive = Path("data/TestDataset.zip")
    if sha256(archive) != protocol["external_archive_sha256"]:
        raise ValueError("External dataset archive differs from the frozen follow-up")
    root = Path(protocol["external_data_dir"])
    known = set()
    original = Path("data/polyp_png")
    for split in ("source_train", "source_val", "target_select", "target_test"):
        known.update(sha256(path) for path in index_images(original / split / "images"))
    datasets = {}
    duplicate_audit = {}
    for name, expected in protocol["external_datasets"].items():
        dataset = SegmentationDataset(root, name, protocol["image_size"], 3)
        if len(dataset) != expected:
            raise ValueError(f"{name} has {len(dataset)} pairs, expected {expected}")
        masks_by_image = {}
        for image, mask in dataset.pairs:
            with Image.open(image) as im, Image.open(mask) as gt:
                im.load()
                gt.load()
                if im.size != gt.size:
                    raise ValueError(f"Image/mask dimensions differ: {image}")
            digest = sha256(image)
            if digest in known:
                raise ValueError(f"Image overlaps another evaluation domain or training split: {image}")
            masks_by_image.setdefault(digest, []).append(sha256(mask))
        known.update(masks_by_image)
        repeated = [masks for masks in masks_by_image.values() if len(masks) > 1]
        duplicate_audit[name] = {
            "unique_images": len(masks_by_image),
            "duplicate_image_groups": len(repeated),
            "extra_image_copies": sum(len(masks) - 1 for masks in repeated),
            "groups_with_different_mask_bytes": sum(len(set(masks)) > 1 for masks in repeated),
        }
        datasets[name] = dataset
    return datasets, duplicate_audit


def preload(dataset, batch_size):
    loaded = []
    offset = 0
    for images, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        if not bool((masks.flatten(1).sum(1) > 0).all()):
            raise ValueError("External mask became empty after the fixed resize")
        ids = [dataset.pairs[offset + index][0].stem for index in range(len(images))]
        loaded.append((ids, images, masks))
        offset += len(images)
    return loaded


def evaluate_checkpoint(checkpoint_path, checkpoint_hash, batches, threshold, device):
    if sha256(checkpoint_path) != checkpoint_hash:
        raise ValueError(f"Checkpoint differs from frozen selection: {checkpoint_path}")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    rows = []
    with torch.inference_mode():
        for ids, images, masks in batches:
            logits = model(images.to(device))
            metrics = per_image_metrics(logits, masks.to(device), threshold)
            rows.extend({"sample_id": sample_id, **values}
                        for sample_id, values in zip(ids, metrics))
    names = ("dice", "iou", "precision", "recall", "foreground_fraction")
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "n_images": len(rows),
        **{f"mean_{name}": statistics.mean(row[name] for row in rows) for name in names},
        "empty_prediction_fraction": statistics.mean(
            row["foreground_fraction"] == 0 for row in rows),
        "per_image": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="followup_protocol.yaml")
    parser.add_argument("--selection", default="runs/real/followup/guarded_selection.json")
    parser.add_argument("--output-dir", default="runs/real/followup/external")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if sha256(args.selection) != protocol["guarded_selection_sha256"]:
        raise ValueError("Guarded selection differs from the frozen follow-up")
    if sha256("protocol.yaml") != protocol["original_protocol_sha256"]:
        raise ValueError("Original experiment protocol differs from the follow-up")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if len(selection["runs"]) != 6:
        raise ValueError("Expected six frozen training runs")
    # Validate all external files and model weights before loading evaluation masks.
    for run in selection["runs"]:
        for method in run["methods"].values():
            if sha256(method["checkpoint"]) != method["checkpoint_sha256"]:
                raise ValueError(f"Changed checkpoint: {method['checkpoint']}")
    datasets, duplicate_audit = audit_external(protocol)
    print("External duplicate audit:", json.dumps(duplicate_audit), flush=True)
    output = Path(args.output_dir)
    lock = {"protocol_sha256": sha256(args.protocol),
            "selection_sha256": sha256(args.selection),
            "external_archive_sha256": protocol["external_archive_sha256"]}
    lock_path = output / "external_lock.json"
    if output.exists() and any(output.iterdir()):
        if not lock_path.exists() or json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise FileExistsError("External output directory has a different frozen input")
    else:
        output.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    device = choose_device(args.device)
    summary = []
    cache = {}
    for domain, dataset in datasets.items():
        print(f"Evaluating {domain}: {len(dataset)} images", flush=True)
        batches = preload(dataset, batch_size=8)
        domain_rows = []
        for run in selection["runs"]:
            method_reports = {}
            for name, chosen in run["methods"].items():
                key = (domain, chosen["checkpoint_sha256"])
                if key not in cache:
                    cache[key] = evaluate_checkpoint(
                        chosen["checkpoint"], chosen["checkpoint_sha256"], batches,
                        float(protocol["prediction_threshold"]), device,
                    )
                method_reports[name] = cache[key]
                report_path = output / domain / run["run"] / f"{name}.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(json.dumps(cache[key], indent=2) + "\n", encoding="utf-8")
            left = {row["sample_id"]: row["dice"] for row in method_reports["source"]["per_image"]}
            if len(left) != len(dataset):
                raise ValueError("Source report has incomplete image IDs")
            for report in method_reports.values():
                if {row["sample_id"] for row in report["per_image"]} != set(left):
                    raise ValueError("Methods evaluated different external images")
            baseline = method_reports["source"]
            old = method_reports["original_stability"]
            guarded = method_reports["guarded"]
            domain_rows.append({
                "run": run["run"], "seed": run["seed"], "augmentation": run["augmentation"],
                "n_images": len(dataset),
                "source_dice": baseline["mean_dice"],
                "original_stability_dice": old["mean_dice"],
                "guarded_dice": guarded["mean_dice"],
                "guarded_minus_source": guarded["mean_dice"] - baseline["mean_dice"],
                "guarded_minus_original": guarded["mean_dice"] - old["mean_dice"],
                "source_empty_fraction": baseline["empty_prediction_fraction"],
                "original_empty_fraction": old["empty_prediction_fraction"],
                "guarded_empty_fraction": guarded["empty_prediction_fraction"],
            })
            print(run["run"], "source", f"{baseline['mean_dice']:.4f}",
                  "original", f"{old['mean_dice']:.4f}",
                  "guarded", f"{guarded['mean_dice']:.4f}", flush=True)
        summary.append({"dataset": domain, "runs": domain_rows})
    (output / "summary.json").write_text(json.dumps({"datasets": summary,
                                                      "duplicate_audit": duplicate_audit}, indent=2) + "\n",
                                         encoding="utf-8")
    print("Saved external evaluation to", output)


if __name__ == "__main__":
    main()
