"""Train one binary U-Net candidate on labeled source data only."""
import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from utils import (
    atomic_torch_save, bce_dice_loss, choose_device, dice_iou_sum,
    move_optimizer_states, read_config, seed_everything,
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


def augment(images, masks):
    # Geometry is applied to both tensors; brightness only to input images.
    if torch.rand(()).item() < 0.5:
        images, masks = images.flip(-1), masks.flip(-1)
    if torch.rand(()).item() < 0.5:
        images, masks = images.flip(-2), masks.flip(-2)
    factor = 0.85 + 0.30 * torch.rand(()).item()
    return (images * factor).clamp(0, 1), masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--data-dir")
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", help="Resume from this run's latest checkpoint")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()

    cfg = read_config(args.config)
    for key in ("data_dir", "run_dir", "epochs"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    seed_everything(int(cfg["seed"]))
    device = choose_device(args.device)
    run_dir = Path(cfg["run_dir"])
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory {run_dir} is not empty; choose a new --run-dir or use --resume"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
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
        for key in ("input_channels", "base_channels", "image_size", "architecture"):
            if old.get(key, "unet") != cfg.get(key, "unet"):
                raise ValueError(f"Checkpoint incompatible: {key}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        move_optimizer_states(optimizer, device)
        if use_amp and saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch = int(saved["epoch"]) + 1
        best = float(saved.get("best_dice", -1))
        if Path(args.resume).parent.resolve() != run_dir.resolve():
            raise ValueError("Resume must use the checkpoint from this run directory")

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
                if cfg.get("augment", True):
                    images, masks = augment(images, masks)
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
