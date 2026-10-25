"""Final evaluation on source validation or held-out target test labels."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.segmentation import SegmentationDataset
from models.unet import build_model
from utils import choose_device, dice_iou_sum


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--selection", help="JSON created by select_model.py")
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=["source_val", "target_test"], default="target_test")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint_path = (
        json.loads(Path(args.selection).read_text(encoding="utf-8"))["selected_checkpoint"]
        if args.selection else args.checkpoint
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    data_dir = args.data_dir or cfg["data_dir"]
    dataset = SegmentationDataset(data_dir, args.split, cfg["image_size"], cfg["input_channels"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dice = iou = count = 0
    with torch.inference_mode():
        for images, masks in loader:
            logits = model(images.to(device))
            d, i, n = dice_iou_sum(logits, masks.to(device))
            dice += d
            iou += i
            count += n
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()), "split": args.split,
        "n_images": count, "mean_dice": dice / count, "mean_iou": iou / count,
        "mask_threshold": 0.5, "device": device.type,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()