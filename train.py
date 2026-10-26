"""Train one binary U-Net candidate on labeled source data only."""
import argparse
import csv
import json
import platform
import subprocess
from importlib import metadata as package_metadata
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from scripts.check_protocol import check_protocol, sha256
from utils import (
    atomic_torch_save, bce_dice_loss, choose_device, dice_iou_sum,
    move_optimizer_states, read_config, seed_everything,
)

AUGMENTATION_PARAMETERS = {
    "a0": {"horizontal_flip_probability": 0.5,
           "vertical_flip_probability": 0.5},
    "a1": {"horizontal_flip_probability": 0.5,
           "vertical_flip_probability": 0.5,
           "photometric_probability": 0.5,
           "brightness_factor": [0.9, 1.1],
           "contrast_factor": [0.9, 1.1],
           "rgb_channel_gain": [0.95, 1.05]},
}
RANDOM_STREAM_POLICY = (
    "model_seed_then_independent_shuffle_seed_plus_1000_geometry_plus_2000_"
    "photometric_plus_3000"
)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    loss_sum = dice_sum = iou_sum = 0.0
    n = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        batch = len(images)
        loss_sum += bce_dice_loss(logits, masks).item() * batch
        d, i, _ = dice_iou_sum(logits, masks)
        dice_sum += d
        iou_sum += i
        n += batch
    return loss_sum / n, dice_sum / n, iou_sum / n


def augment(images, masks, mode, geometry_rng=None, photometric_rng=None):
    """A0: paired flips. A1: the same flips plus mild image-only color changes."""
    if mode not in AUGMENTATION_PARAMETERS:
        raise ValueError(f"Unknown augmentation: {mode}")
    parameters = AUGMENTATION_PARAMETERS[mode]
    if torch.rand((), generator=geometry_rng).item() < parameters["horizontal_flip_probability"]:
        images, masks = images.flip(-1), masks.flip(-1)
    if torch.rand((), generator=geometry_rng).item() < parameters["vertical_flip_probability"]:
        images, masks = images.flip(-2), masks.flip(-2)
    if mode == "a1" and torch.rand((), generator=photometric_rng).item() < parameters["photometric_probability"]:
        brightness = parameters["brightness_factor"][0] + (
            parameters["brightness_factor"][1] - parameters["brightness_factor"][0]
        ) * torch.rand((), generator=photometric_rng).item()
        contrast = parameters["contrast_factor"][0] + (
            parameters["contrast_factor"][1] - parameters["contrast_factor"][0]
        ) * torch.rand((), generator=photometric_rng).item()
        mean = images.mean(dim=(-2, -1), keepdim=True)
        images = (images - mean) * contrast + mean
        images = images * brightness
        if images.shape[1] == 3:
            low, high = parameters["rgb_channel_gain"]
            color = low + (high - low) * torch.rand(3, generator=photometric_rng,
                                                   device=images.device)
            images = images * color.view(1, 3, 1, 1)
        images = images.clamp(0, 1)
    return images, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--data-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Resume from this run's latest checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--augmentation", choices=["a0", "a1"])
    parser.add_argument("--protocol", default="protocol.yaml")
    args = parser.parse_args()

    cfg = read_config(args.config)
    for key in ("data_dir", "run_dir", "epochs", "seed", "augmentation"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    cfg["augmentation"] = cfg.get("augmentation", "a0")
    if cfg["augmentation"] not in ("a0", "a1"):
        raise ValueError("augmentation must be a0 or a1")
    if int(cfg["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    demo = Path(cfg["data_dir"]).resolve() == Path("data/demo").resolve()
    protocol_info = None
    if not demo:
        protocol_info = check_protocol(args.protocol, cfg["data_dir"])
        protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
        if int(cfg["epochs"]) != int(protocol["epochs"]):
            raise ValueError("Real training epochs differ from frozen protocol")
        if int(cfg["seed"]) not in protocol["seeds"]:
            raise ValueError("Training seed differs from frozen protocol")
        if cfg["augmentation"] not in protocol["augmentations"]:
            raise ValueError("Augmentation differs from frozen protocol")
        if protocol["augmentation_parameters"] != AUGMENTATION_PARAMETERS:
            raise ValueError("Augmentation parameters differ from the frozen protocol")
        if protocol["random_streams"] != RANDOM_STREAM_POLICY:
            raise ValueError("Random-stream policy differs from the frozen protocol")
    seed_everything(int(cfg["seed"]))
    shuffle_rng = torch.Generator().manual_seed(int(cfg["seed"]) + 1000)
    geometry_rng = torch.Generator().manual_seed(int(cfg["seed"]) + 2000)
    photometric_rng = torch.Generator().manual_seed(int(cfg["seed"]) + 3000)
    device = choose_device(args.device)
    run_dir = Path(cfg["run_dir"])
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory {run_dir} is not empty; choose a new --run-dir or use --resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "resolved_config.yaml"
    if args.resume and config_path.exists():
        previous_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if previous_cfg != cfg:
            raise ValueError("Resume configuration differs from the recorded run configuration")
    else:
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=True), encoding="utf-8")
    metadata_path = run_dir / "run_metadata.json"
    if not args.resume:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=False)
        metadata = {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "dependencies": {package: package_metadata.version(package)
                             for package in ("numpy", "Pillow", "PyYAML", "tqdm")},
            "device": device.type,
            "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            "protocol_sha256": sha256(args.protocol) if not demo else None,
            "split_manifest_sha256": protocol_info["manifest_sha256"] if protocol_info else None,
            "augmentation_parameters": AUGMENTATION_PARAMETERS[cfg["augmentation"]],
            "random_streams": RANDOM_STREAM_POLICY,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PyTorch={torch.__version__}; device={device}; source={cfg['data_dir']}")

    train_set = SegmentationDataset(
        cfg["data_dir"], "source_train", cfg["image_size"], cfg["input_channels"]
    )
    val_set = SegmentationDataset(
        cfg["data_dir"], "source_val", cfg["image_size"], cfg["input_channels"]
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=device.type == "cuda",
        generator=shuffle_rng,
    )
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    use_amp = bool(cfg.get("amp_cuda", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best = -1.0

    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        old = saved["config"]
        if old != cfg:
            raise ValueError("Checkpoint configuration differs from this run")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        move_optimizer_states(optimizer, device)
        if use_amp and saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get("best_dice", -1))
        if Path(args.resume).parent.resolve() != run_dir.resolve():
            raise ValueError("Resume must use the checkpoint from this run directory")
        if saved.get("cpu_rng_state") is not None:
            torch.set_rng_state(saved["cpu_rng_state"])
        if device.type == "cuda" and saved.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng_states"])
        for generator, key in ((shuffle_rng, "shuffle_rng_state"),
                               (geometry_rng, "geometry_rng_state"),
                               (photometric_rng, "photometric_rng_state")):
            if saved.get(key) is not None:
                generator.set_state(saved[key])

    log_path = run_dir / "metrics.csv"
    append = bool(args.resume and log_path.exists())
    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(["epoch", "train_loss", "source_val_loss", "source_val_dice",
                             "source_val_iou", "device"])
        for epoch in range(start_epoch, int(cfg["epochs"])):
            model.train()
            loss_sum = 0.0
            count = 0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}"):
                images, masks = augment(images, masks, cfg["augmentation"],
                                        geometry_rng, photometric_rng)
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    loss = bce_dice_loss(model(images), masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += loss.item() * len(images)
                count += len(images)
            val_loss, val_dice, val_iou = validate(model, val_loader, device)
            improved = val_dice > best
            best = max(best, val_dice)
            state = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(), "config": cfg,
                "source_val_dice": val_dice, "best_dice": best,
                "torch_version": str(torch.__version__),
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                "shuffle_rng_state": shuffle_rng.get_state(),
                "geometry_rng_state": geometry_rng.get_state(),
                "photometric_rng_state": photometric_rng.get_state(),
            }
            atomic_torch_save(state, run_dir / "checkpoints" / f"epoch_{epoch + 1:03d}.pt")
            atomic_torch_save(state, run_dir / "latest.pt")
            if improved:
                atomic_torch_save(state, run_dir / "best_source.pt")
            writer.writerow([epoch + 1, loss_sum / count, val_loss, val_dice, val_iou,
                             device.type])
            handle.flush()
            print(f"Epoch {epoch + 1}: source Dice={val_dice:.4f}, IoU={val_iou:.4f}")
    print("Outputs:", run_dir)


if __name__ == "__main__":
    main()
