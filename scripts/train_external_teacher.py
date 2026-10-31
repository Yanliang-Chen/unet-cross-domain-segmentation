"""Matched heterogeneous-teacher and homogeneous-teacher source-anchored arms."""
import copy
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import yaml

from datasets.segmentation import SegmentationDataset, load_image
from models.unet import build_model
from models.reciprocal_distillation import reciprocal_loss
from scripts.check_protocol import sha256
from scripts.artifacts import update_teacher
from train import augment, validate
from utils import seed_everything, choose_device, bce_dice_loss, atomic_torch_save


class CachedTargets(Dataset):
    def __init__(self, files, arm):
        self.files, self.arm = files, arm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        row = self.files[index]
        data = torch.load(row["cache"], map_location="cpu", weights_only=True)
        source = data["source_probability"].float()
        foundation = data["foundation_probability"].float() if self.arm == "external" else source
        return load_image(row["image"], 256, 3), source, foundation, data["valid"]


def train_one(run, arm, p, digest, root, device):
    if arm not in ("external", "self"):
        raise ValueError("Only the retained matched external/self arms are supported")
    seed = run["seed"]
    seed_everything(seed+10000)
    source_rng = torch.Generator().manual_seed(seed+11000)
    target_rng = torch.Generator().manual_seed(seed+12000)
    geometry_rng = torch.Generator().manual_seed(seed+13000)
    photo_rng = torch.Generator().manual_seed(seed+14000)
    noise_rng = torch.Generator().manual_seed(seed+15000)
    source = run["source"]
    if sha256(source["checkpoint"]) != source["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    saved = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    cfg = saved["config"]
    if cfg["seed"] != seed or cfg["augmentation"] != run["augmentation"] or cfg["image_size"] != 256:
        raise ValueError("Wrong source checkpoint identity")
    source_data = SegmentationDataset(p["source_data"], "source_train", 256, 3)
    validation_data = SegmentationDataset(p["source_data"], "source_val", 256, 3)
    if len(source_data) != p["source_train_count"] or len(validation_data) != p["source_val_count"]:
        raise ValueError("Source split changed")
    source_loader = DataLoader(source_data, batch_size=p["batch_size"], shuffle=True, generator=source_rng)
    val_loader = DataLoader(validation_data, batch_size=p["batch_size"])
    target_loader = (DataLoader(CachedTargets(run["files"], arm), batch_size=p["batch_size"],
                               shuffle=True, generator=target_rng) if arm != "source" else None)
    student = build_model(cfg).to(device)
    student.load_state_dict(saved["model_state_dict"])
    teacher = copy.deepcopy(student)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(p["learning_rate"]),
                                  weight_decay=float(p["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda")
    directory = root/arm/run["run"]
    directory.mkdir(parents=True)
    best = -1
    updates = 0
    with (directory/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "source_loss", "distillation_loss", "mean_agreement", "valid_fraction",
                         "source_val_dice", "source_val_iou"])
        for epoch in range(1, p["epochs"]+1):
            student.train()
            for module in student.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            teacher.eval()
            target_iter = iter(target_loader) if target_loader is not None else None
            sums = [0.,0.,0.,0.]
            for images, masks in source_loader:
                if target_iter is not None:
                    try:
                        x, ps, pf, valid = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        x, ps, pf, valid = next(target_iter)
                    noise = torch.randn(x.shape, generator=noise_rng)*float(p["target_noise_sigma"])
                    x = (x+noise).clamp(0,1).to(device)
                    ps, pf, valid = ps.to(device), pf.to(device), valid.to(device)
                images, masks = augment(images, masks, run["augmentation"], geometry_rng, photo_rng)
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda"):
                    supervised = bce_dice_loss(student(images), masks)
                    if target_loader is not None:
                        distill, diagnostics = reciprocal_loss(student(x), ps, pf, valid, p)
                    else:
                        distill = supervised.new_zeros(())
                        diagnostics = {"agreement":0., "valid_fraction":0.}
                    loss = supervised+float(p["target_loss_weight"])*distill
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite objective")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                update_teacher(teacher, student, float(p["ema_decay"]))
                updates += 1
                for i, value in enumerate((float(supervised.detach()), float(distill.detach()), diagnostics["agreement"], diagnostics["valid_fraction"])):
                    sums[i] += value
            _, dice, iou = validate(teacher, val_loader, device, cfg)
            state = {"config": cfg, "model_state_dict": teacher.state_dict(), "epoch": epoch,
                     "source_val_dice": dice, "adaptation_protocol_sha256": digest,
                     "source_checkpoint_sha256": source["checkpoint_sha256"], "arm": arm}
            atomic_torch_save(state, directory/f"epoch_{epoch:03d}.pt")
            if dice > best:
                best = dice
                atomic_torch_save(state, directory/"best_source.pt")
            writer.writerow([epoch, *(x/len(source_loader) for x in sums), dice, iou])
            stream.flush()
            print(arm, run["run"], "epoch", epoch, "source_val", round(dice,4),
                  "target_loss", round(sums[1]/len(source_loader),4), flush=True)
    if updates != p["gradient_updates_per_run"]:
        raise ValueError("Wrong update budget")
    chosen = torch.load(directory/"best_source.pt", map_location="cpu", weights_only=True)
    original = saved["model_state_dict"]
    if any(not torch.equal(tensor, original[key]) for key, tensor in chosen["model_state_dict"].items()
           if key.endswith(("running_mean", "running_var", "num_batches_tracked"))):
        raise ValueError("Frozen BN buffers changed")
    return {"run": run["run"], "seed": seed, "augmentation": run["augmentation"], "arm": arm,
            "source": source, "best_epoch": chosen["epoch"], "source_val_dice": chosen["source_val_dice"],
            "best_checkpoint": str((directory/"best_source.pt").resolve()),
            "best_checkpoint_sha256": sha256(directory/"best_source.pt"), "updates": updates}


def main():
    protocol = Path("external_teacher_protocol.yaml")
    p = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    root = Path("runs/real/external_teacher")
    manifest_path = root/"targets/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256(protocol)
    if manifest["protocol_sha256"] != digest:
        raise ValueError("Teacher generation protocol changed")
    expected = {f"{a}_seed{s}" for a in p["augmentations"] for s in p["seeds"]}
    if {r["run"] for r in manifest["runs"]} != expected or len(manifest["runs"]) != len(expected):
        raise ValueError("Incomplete teacher set")
    for arm in p["arms"]:
        if (root/arm).exists():
            raise FileExistsError(root/arm)
    for run in manifest["runs"]:
        if len(run["files"]) != p["target_count"]:
            raise ValueError("Incomplete target cache")
        for row in run["files"]:
            if sha256(row["image"]) != row["image_sha256"] or sha256(row["cache"]) != row["cache_sha256"]:
                raise ValueError("Teacher cache changed")
    device = choose_device("cuda")
    torch.set_num_threads(4)
    results = []
    for run in manifest["runs"]:
        for arm in p["arms"]:
            results.append(train_one(run, arm, p, digest, root, device))
    (root/"adaptation.json").write_text(json.dumps({"protocol_sha256": digest,
        "teacher_manifest_sha256": sha256(manifest_path), "runs": results}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
