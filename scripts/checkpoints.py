"""Load a frozen checkpoint after checking its identity."""
import torch
from models.unet import build_model
from scripts.check_protocol import sha256


def checkpoint_model(path, expected_hash, expected_seed, expected_augmentation,
                     device, expected_loss="bce_dice"):
    if sha256(path) != expected_hash:
        raise ValueError(f"Checkpoint differs from frozen input: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state["config"]
    if (int(cfg["seed"]) != expected_seed or cfg["augmentation"] != expected_augmentation
            or cfg.get("loss", "bce_dice") != expected_loss):
        raise ValueError(f"Checkpoint identity differs: {path}")
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model, int(cfg["image_size"])
