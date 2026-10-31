"""Final evaluation on source validation or held-out target test labels."""
import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from scripts.check_protocol import check_protocol, sha256
from utils import choose_device, per_image_metrics


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--selection", help="JSON created by select_model.py")
    parser.add_argument("--method", choices=["source"],
                        default="source", help="Method to read from --selection")
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=["source_val", "target_test"], default="target_test")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--protocol", default="protocol.yaml")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Confirm that all runs, selection rules and outputs are frozen")
    parser.add_argument("--prevalidated-manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if args.split == "target_test" and not args.final_evaluation:
        parser.error("target_test requires --final-evaluation after the protocol and selections are frozen")
    device = choose_device(args.device)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8")) if args.selection else None
    checkpoint_path = (
        selection["methods"][args.method] if selection else args.checkpoint
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    data_dir = args.data_dir or cfg["data_dir"]
    demo = Path(data_dir).resolve() == Path("data/demo").resolve()
    protocol_info = None
    if args.split == "target_test" and not demo:
        if args.prevalidated_manifest_sha256:
            actual = sha256(Path(data_dir) / "split_manifest.csv")
            locked = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))[
                "split_manifest_sha256"]
            if actual != args.prevalidated_manifest_sha256 or actual != locked:
                raise ValueError("Prevalidated manifest hash differs from the current locked split")
            protocol_info = {"manifest_sha256": actual}
        else:
            protocol_info = check_protocol(args.protocol, data_dir)
        checkpoint_location = Path(checkpoint_path).resolve()
        run_dir = (checkpoint_location.parent.parent if checkpoint_location.parent.name == "checkpoints"
                   else checkpoint_location.parent)
        metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if metadata["protocol_sha256"] != sha256(args.protocol) or (
                metadata["split_manifest_sha256"] != protocol_info["manifest_sha256"]):
            raise ValueError("Checkpoint run metadata differs from the frozen protocol/data split")
    threshold = 0.5
    if not demo and Path(args.protocol).exists():
        threshold = float(yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))[
            "prediction_threshold"])
    dataset = SegmentationDataset(data_dir, args.split, cfg["image_size"], cfg["input_channels"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    groups = {}
    manifest = Path(data_dir) / "split_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            groups = {row["sample_id"]: row["group_id"] for row in csv.DictReader(handle)
                      if row["split"] == args.split}
    per_image = []
    offset = 0
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            batch_metrics = per_image_metrics(logits, masks.to(device), threshold)
            for index, metrics in enumerate(batch_metrics):
                sample_id = dataset.pairs[offset + index][0].stem
                per_image.append({"sample_id": sample_id, "group_id": groups.get(sample_id, ""),
                                  **metrics})
            offset += len(batch_metrics)
    names = ("dice", "iou", "precision", "recall", "foreground_fraction")
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "split": args.split,
        "method": args.method if selection else "direct_checkpoint",
        "seed": int(cfg["seed"]), "augmentation": cfg.get("augmentation", "a0"),
        "n_images": len(per_image),
        **{f"mean_{name}": sum(row[name] for row in per_image) / len(per_image)
           for name in names},
        "empty_prediction_fraction": sum(row["foreground_fraction"] == 0 for row in per_image)
                                      / len(per_image),
        "mask_threshold": threshold, "device": device.type,
        "split_manifest_sha256": protocol_info["manifest_sha256"] if protocol_info else None,
        "per_image": per_image,
    }
    print(json.dumps({key: value for key, value in result.items() if key != "per_image"}, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=per_image[0].keys())
            writer.writeheader()
            writer.writerows(per_image)


if __name__ == "__main__":
    main()
