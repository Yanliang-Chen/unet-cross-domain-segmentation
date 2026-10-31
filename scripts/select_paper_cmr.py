"""Rank frozen checkpoints with foreground-only consistency on unlabeled target images."""
import argparse
import json
import statistics
from pathlib import Path

import torch
import yaml

from datasets.segmentation import ImagesOnlyDataset
from models.unet import build_model
from scripts.check_protocol import sha256
from utils import choose_device


def foreground_iou(reference, perturbed, threshold=0.5, empty_score=0.0):
    """CMR-NHD for binary segmentation; empty/empty has no foreground evidence."""
    a = reference > threshold
    b = perturbed > threshold
    intersection = (a & b).flatten(1).sum(1).float()
    union = (a | b).flatten(1).sum(1).float()
    return torch.where(union > 0, intersection / union.clamp_min(1), empty_score)


def load_unlabeled_batches(protocol):
    directory = Path(protocol["target_select_images"])
    if directory.resolve() != (Path("data/polyp_png") / "target_select" / "images").resolve():
        raise ValueError("CMR may only read the original unlabeled target selection images")
    dataset = ImagesOnlyDataset(directory, 256, 3)
    if len(dataset) != protocol["expected_target_select_count"]:
        raise ValueError("Target selection image count differs from the protocol")
    generator = torch.Generator().manual_seed(int(protocol["cmr_noise_seed"]))
    sigma = float(protocol["cmr_sigma"])
    batches = []
    for start in range(0, len(dataset), 8):
        images = torch.stack([dataset[i] for i in range(start, min(start + 8, len(dataset)))])
        noise = torch.randn(images.shape, generator=generator) * sigma
        batches.append((images, (images + noise).clamp(0, 1)))
    return batches


@torch.inference_mode()
def cmr_score(path, batches, device, threshold, empty_score):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(saved["config"]).to(device)
    model.load_state_dict(saved["model_state_dict"])
    model.eval()
    values = []
    empty_count = 0
    for images, noisy in batches:
        plain = model(images.to(device)).sigmoid()
        changed = model(noisy.to(device)).sigmoid()
        values.extend(foreground_iou(plain, changed, threshold, empty_score).cpu().tolist())
        empty_count += int((plain > threshold).flatten(1).sum(1).eq(0).sum().item())
    del model
    return statistics.median(values), empty_count / len(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="paper_selection_protocol.yaml")
    parser.add_argument("--selection-dir", default="runs/real/selections")
    parser.add_argument("--output", default="runs/real/paper_selection/cmr_selection.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if (protocol["cmr_metric"] != "foreground_restricted_nhd_iou"
            or protocol["cmr_perturbation"] != "additive_gaussian_noise"
            or protocol["cmr_per_image_aggregation"] != "median"):
        raise ValueError("Unsupported CMR protocol")
    if sha256("protocol.yaml") != protocol["original_protocol_sha256"]:
        raise ValueError("Original six-run protocol changed")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Selection already exists: {output}")
    batches = load_unlabeled_batches(protocol)
    device = choose_device(args.device)
    runs = []
    for augmentation in protocol["augmentations"]:
        for seed in protocol["seeds"]:
            name = f"{augmentation}_seed{seed}"
            path = Path(args.selection_dir) / f"{name}.json"
            if sha256(path) != protocol["original_selection_sha256"][name]:
                raise ValueError(f"Original selection changed: {name}")
            original = json.loads(path.read_text(encoding="utf-8"))
            rows = original["candidates"]
            if len(rows) != 30 or sorted(row["epoch"] for row in rows) != list(range(1, 31)):
                raise ValueError(f"Incomplete candidate pool: {name}")
            source = next(row for row in rows if row["checkpoint"] == original["methods"]["source"])
            eligible = [row for row in rows
                        if row["source_val_dice"] >= source["source_val_dice"]
                        - float(protocol["source_dice_tolerance"])
                        and row["empty_prediction_fraction"] <= source["empty_prediction_fraction"]
                        + float(protocol["empty_fraction_margin"])]
            if not eligible:
                raise ValueError(f"No eligible checkpoint: {name}")
            scored = []
            for row in eligible:
                checkpoint = Path(row["checkpoint"])
                if not checkpoint.is_file() or checkpoint.resolve().parent != (
                    Path(original["run_dir"]) / "checkpoints").resolve():
                    raise ValueError(f"Checkpoint outside original run: {checkpoint}")
                score, empty = cmr_score(checkpoint, batches, device,
                                         float(protocol["prediction_threshold"]),
                                         float(protocol["empty_union_score"]))
                scored.append({"checkpoint": str(checkpoint.resolve()), "epoch": row["epoch"],
                               "source_val_dice": row["source_val_dice"], "cmr_nhd": score,
                               "empty_prediction_fraction": empty})
            winner = min(scored, key=lambda row: (-row["cmr_nhd"],
                                                     -row["source_val_dice"], row["checkpoint"]))
            choices = {}
            for key, row in (("source", source), ("cmr", winner)):
                checkpoint = Path(row["checkpoint"])
                choices[key] = {"checkpoint": str(checkpoint.resolve()),
                                "checkpoint_sha256": sha256(checkpoint), "epoch": row["epoch"]}
            runs.append({"run": name, "augmentation": augmentation, "seed": seed,
                         "eligible_count": len(eligible), "candidates": scored, "methods": choices})
            print(name, "eligible", len(eligible), "source", source["epoch"],
                  "cmr", winner["epoch"], f"score={winner['cmr_nhd']:.4f}", flush=True)
    result = {"status": protocol["status"], "protocol_sha256": sha256(args.protocol),
              "original_protocol_sha256": protocol["original_protocol_sha256"],
              "target_select_images": str(Path(protocol["target_select_images"]).resolve()),
              "runs": runs}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Saved", output, "sha256", sha256(output), flush=True)


if __name__ == "__main__":
    main()
