"""Evaluate a binary segmentation checkpoint."""
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.segmentation import SegmentationDataset
from models.unet import UNet
from utils import choose_device,dice_iou_sum

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--data-dir")
    ap.add_argument("--split",choices=["val","test"],default="test")
    ap.add_argument("--device",choices=["auto","cuda","mps","cpu"],default="auto")
    args=ap.parse_args()
    device=choose_device(args.device)
    ckpt=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    cfg=ckpt["config"]
    data_dir=args.data_dir or cfg["data_dir"]
    ds=SegmentationDataset(data_dir,args.split,cfg["image_size"],cfg["input_channels"])
    dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=False,num_workers=0)
    model=UNet(cfg["input_channels"],cfg["base_channels"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    dice=iou=0.0
    count=0
    with torch.inference_mode():
        for images,masks in dl:
            logits=model(images.to(device))
            d,i,n=dice_iou_sum(logits,masks.to(device))
            dice+=d
            iou+=i
            count+=n
    print(f"{args.split}: {count} images; device={device}; Dice={dice/count:.4f}; IoU={iou/count:.4f}")

if __name__=="__main__":
    main()
