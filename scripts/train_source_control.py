"""Retained strong source-only continuation recipe; no target adaptation."""
import argparse
import copy
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from scripts.check_protocol import sha256
from scripts.artifacts import update_teacher
from train import augment, validate
from utils import atomic_torch_save, bce_dice_loss, choose_device, seed_everything


def train_one(run, protocol, protocol_hash, output_root, device):
    name = run["run"]
    seed = int(run["seed"])
    augmentation = run["augmentation"]
    source = run["methods"]["source"]
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError(f"Source checkpoint changed: {name}")
    saved = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    cfg = saved["config"]
    if (int(cfg["seed"]) != seed or cfg["augmentation"] != augmentation
            or int(cfg["image_size"]) != 256):
        raise ValueError(f"Source checkpoint identity changed: {name}")
    seed_everything(seed + 10000)
    source_shuffle = torch.Generator().manual_seed(seed + 11000)
    geometry_rng = torch.Generator().manual_seed(seed + 13000)
    photometric_rng = torch.Generator().manual_seed(seed + 14000)
    train_data = SegmentationDataset(protocol["data_dir"], "source_train", 256, 3)
    val_data = SegmentationDataset(protocol["data_dir"], "source_val", 256, 3)
    if len(train_data) != protocol["source_train_count"] or len(val_data) != protocol["source_val_count"]:
        raise ValueError("Source split changed")
    train_loader = DataLoader(train_data, batch_size=protocol["batch_size"], shuffle=True,
                              generator=source_shuffle, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=protocol["batch_size"], shuffle=False)
    student = build_model(cfg).to(device)
    student.load_state_dict(saved["model_state_dict"])
    teacher = copy.deepcopy(student).to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(protocol["learning_rate"]),
                                  weight_decay=float(protocol["weight_decay"]))
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best_dice = -1.0
    run_dir = output_root / name
    run_dir.mkdir(parents=True)
    log_path = run_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "source_loss", "target_consistency_loss",
                         "source_val_dice", "source_val_iou"])
        for epoch in range(1, int(protocol["epochs"]) + 1):
            student.train()
            if protocol.get("freeze_bn_statistics", False):
                for module in student.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.eval()
            teacher.eval()
            source_total = consistency_total = 0.0
            examples = 0
            for source_images, source_masks in train_loader:
                source_images, source_masks = augment(source_images, source_masks,
                                                       augmentation, geometry_rng, photometric_rng)
                source_images = source_images.to(device)
                source_masks = source_masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    source_loss = bce_dice_loss(student(source_images), source_masks)
                    consistency = source_loss.new_zeros(())
                    loss = source_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                update_teacher(teacher, student, float(protocol["teacher_ema_decay"]))
                source_total += source_loss.item() * len(source_images)
                consistency_total += consistency.item() * len(source_images)
                examples += len(source_images)
            _, val_dice, val_iou = validate(teacher, val_loader, device, cfg)
            state = {"config": cfg, "model_state_dict": teacher.state_dict(),
                     "epoch": epoch, "source_val_dice": val_dice,
                     "source_checkpoint_sha256": source["checkpoint_sha256"],
                     "adaptation_protocol_sha256": protocol_hash}
            atomic_torch_save(state, run_dir / f"epoch_{epoch:03d}.pt")
            if val_dice > best_dice:
                best_dice = val_dice
                atomic_torch_save(state, run_dir / "best_source.pt")
            writer.writerow([epoch, source_total / examples, consistency_total / examples,
                             val_dice, val_iou])
            log.flush()
            print(name, "epoch", epoch, "source_val_dice", f"{val_dice:.4f}",
                  "consistency", f"{consistency_total / examples:.5f}", flush=True)
    best = torch.load(run_dir / "best_source.pt", map_location="cpu", weights_only=True)
    return {"run": name, "seed": seed, "augmentation": augmentation,
            "source": source, "best_epoch": best["epoch"],
            "best_source_val_dice": best["source_val_dice"],
            "best_checkpoint": str((run_dir / "best_source.pt").resolve()),
            "best_checkpoint_sha256": sha256(run_dir / "best_source.pt"),
            "final_checkpoint": str((run_dir / f"epoch_{protocol['epochs']:03d}.pt").resolve()),
            "final_checkpoint_sha256": sha256(run_dir / f"epoch_{protocol['epochs']:03d}.pt")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="polypgen_source_control_protocol.yaml")
    parser.add_argument("--selection", default="runs/real/followup/guarded_selection.json")
    parser.add_argument("--output-dir", default="runs/real/source_control_clean")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if float(protocol["target_consistency_weight"]) != 0 or not protocol["freeze_bn_statistics"]:
        raise ValueError("Only the fixed-BN, zero-target-loss source baseline is retained")
    if sha256(args.selection) != protocol["source_selection_sha256"]:
        raise ValueError("Source checkpoint selection changed")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    expected = {f"{augmentation}_seed{seed}" for augmentation in protocol["augmentations"]
                for seed in protocol["seeds"]}
    if {run["run"] for run in selection["runs"]} != expected or len(selection["runs"]) != len(expected):
        raise ValueError("Incomplete source selection")
    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Source continuation output already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    rows = [train_one(run, protocol, sha256(args.protocol), output_root, device)
            for run in selection["runs"]]
    manifest = output_root / "adaptation.json"
    manifest.write_text(json.dumps({"status": protocol["status"],
                                    "protocol_sha256": sha256(args.protocol),
                                    "source_selection_sha256": sha256(args.selection),
                                    "runs": rows}, indent=2) + "\n", encoding="utf-8")
    print("Saved", manifest, "sha256", sha256(manifest), flush=True)


if __name__ == "__main__":
    main()
