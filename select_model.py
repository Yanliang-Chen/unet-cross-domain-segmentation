"""Select a checkpoint using source validation labels and unlabeled target images.

This process never constructs a mask dataset or reads target test images.
The stability proxy is a hypothesis to be evaluated, not a proven Dice estimator.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.segmentation import ImagesOnlyDataset
from models.unet import build_model
from utils import choose_device


@torch.inference_mode()
def target_diagnostics(model, loader, device):
    model.eval()
    totals = {"target_instability": 0.0, "mean_entropy": 0.0,
              "mean_foreground_fraction": 0.0, "empty_prediction_fraction": 0.0}
    count = 0
    for images in loader:
        images = images.to(device)
        reference = model(images).sigmoid()
        # Inverse transform returns flipped predictions to original coordinates.
        flipped = model(images.flip(-1)).sigmoid().flip(-1)
        bright = model((images * 0.9 + 0.05).clamp(0, 1)).sigmoid()
        # Mean absolute probability difference, averaged per image.
        per_image = (
            (reference - flipped).abs().mean(dim=(1, 2, 3))
            + (reference - bright).abs().mean(dim=(1, 2, 3))
        ) / 2
        probability = reference.clamp(1e-7, 1 - 1e-7)
        entropy = -(probability * probability.log()
                    + (1 - probability) * (1 - probability).log()).mean(dim=(1, 2, 3))
        foreground = (reference > 0.5).float().mean(dim=(1, 2, 3))
        totals["target_instability"] += per_image.sum().item()
        totals["mean_entropy"] += entropy.sum().item()
        totals["mean_foreground_fraction"] += foreground.sum().item()
        totals["empty_prediction_fraction"] += (foreground == 0).sum().item()
        count += len(images)
    return {key: value / count for key, value in totals.items()}


def select_rows(rows, tolerance):
    """First retain candidates close to the best source Dice, then minimize instability."""
    best_source = max(row["source_val_dice"] for row in rows)
    eligible = [row for row in rows
                if row["source_val_dice"] >= best_source - tolerance]
    stability = min(eligible, key=lambda row: (
        row["target_instability"], -row["source_val_dice"], row["checkpoint"]
    ))
    confidence = min(eligible, key=lambda row: (
        row["mean_entropy"], -row["source_val_dice"], row["checkpoint"]
    ))
    source = min(rows, key=lambda row: (-row["source_val_dice"], row["checkpoint"]))
    for row in rows:
        row["eligible"] = row in eligible
        row["selected_by"] = ",".join(
            name for name, winner in (("source", source), ("stability", stability),
                                      ("confidence", confidence)) if row is winner
        )
    return {"source": source, "stability": stability, "confidence": confidence}, best_source


def require_target_selection_images(target, cfg):
    expected = Path(cfg["data_dir"]) / "target_select" / "images"
    if Path(target).resolve() != expected.resolve():
        raise ValueError(f"Target images must be {expected}, not {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Run directories containing candidate checkpoints")
    parser.add_argument("--target-images", default="data/polyp_png/target_select/images")
    parser.add_argument("--output", default="runs/selection.json")
    parser.add_argument("--tolerance", type=float, default=0.03,
                        help="Maximum permitted source Dice drop from best candidate")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    if len(args.runs) != 1:
        parser.error("Select one training run at a time; the protocol compares methods within a run")
    if not 0 <= args.tolerance <= 1:
        parser.error("--tolerance must be between 0 and 1")

    device = choose_device(args.device)
    paths = sorted({p.resolve() for run in args.runs
                    for p in (Path(run) / "checkpoints").glob("epoch_*.pt")})
    if not paths:
        parser.error("No per-epoch checkpoints found under --runs")
    target = Path(args.target_images)
    rows = []
    run_identity = None
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        cfg = checkpoint["config"]
        if "source_val_dice" not in checkpoint:
            raise ValueError(f"Checkpoint has no source validation Dice: {path}")
        require_target_selection_images(target, cfg)
        identity = (cfg["data_dir"], cfg["seed"], cfg.get("augmentation", "a0"))
        if run_identity is not None and identity != run_identity:
            raise ValueError("Candidates in one run have inconsistent data, seed or augmentation")
        run_identity = identity
        dataset = ImagesOnlyDataset(target, cfg["image_size"], cfg["input_channels"])
        loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False)
        model = build_model(cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        diagnostics = target_diagnostics(model, loader, device)
        row = {
            "checkpoint": str(path), "architecture": cfg.get("architecture", "unet"),
            "seed": int(cfg["seed"]), "augmentation": cfg.get("augmentation", "a0"),
            "epoch": int(checkpoint["epoch"]) + 1,
            "source_val_dice": float(checkpoint["source_val_dice"]),
            **diagnostics,
            "target_image_count": len(dataset),
        }
        rows.append(row)
        print(f"{row['architecture']} epoch {row['epoch']}: "
              f"source Dice={row['source_val_dice']:.4f}; "
              f"instability={diagnostics['target_instability']:.5f}")
    selected, best_source = select_rows(rows, args.tolerance)
    result = {
        "method": "min_target_instability_with_source_dice_guardrail",
        "tolerance": args.tolerance, "best_source_val_dice": best_source,
        "target_images": str(target.resolve()),
        "run_dir": str(Path(args.runs[0]).resolve()),
        "seed": run_identity[1], "augmentation": run_identity[2],
        "selected_checkpoint": selected["stability"]["checkpoint"],
        "methods": {name: row["checkpoint"] for name, row in selected.items()},
        "candidates": rows,
        "warning": "Unsupervised stability is a heuristic and may favor trivial masks.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Selected {selected['stability']['checkpoint']} (source Dice >= "
          f"{best_source - args.tolerance:.4f}); saved {output}")


if __name__ == "__main__":
    main()
