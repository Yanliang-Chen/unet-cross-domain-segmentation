"""Cross-platform CUDA / MPS binary U-Net training."""
import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.segmentation import SegmentationDataset
from models.unet import UNet
from utils import (atomic_torch_save,bce_dice_loss,choose_device,dice_iou_sum,
                   move_optimizer_states,read_config,seed_everything)

def evaluate(model,loader,device):
    model.eval()
    loss_sum=dice_sum=iou_sum=0.0
    n=0
    with torch.inference_mode():
        for images,masks in loader:
            images,masks=images.to(device),masks.to(device)
            logits=model(images)
            bs=len(images)
            loss_sum+=bce_dice_loss(logits,masks).item()*bs
            d,i,_=dice_iou_sum(logits,masks)
            dice_sum+=d
            iou_sum+=i
            n+=bs
    return loss_sum/n,dice_sum/n,iou_sum/n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--device",choices=["auto","cuda","mps","cpu"],default="auto")
    ap.add_argument("--data-dir")
    ap.add_argument("--run-dir")
    ap.add_argument("--resume")
    ap.add_argument("--epochs",type=int)
    args=ap.parse_args()

    cfg=read_config(args.config)
    if args.data_dir is not None: cfg["data_dir"]=args.data_dir
    if args.run_dir is not None: cfg["run_dir"]=args.run_dir
    if args.epochs is not None: cfg["epochs"]=args.epochs
    seed_everything(int(cfg["seed"]))
    device=choose_device(args.device)
    run_dir=Path(cfg["run_dir"])
    run_dir.mkdir(parents=True,exist_ok=True)
    print(f"PyTorch={torch.__version__}; device={device}; data={cfg['data_dir']}")

    train_set=SegmentationDataset(cfg["data_dir"],"train",cfg["image_size"],cfg["input_channels"])
    val_set=SegmentationDataset(cfg["data_dir"],"val",cfg["image_size"],cfg["input_channels"])
    train_dl=DataLoader(train_set,batch_size=cfg["batch_size"],shuffle=True,
                        num_workers=cfg["num_workers"],pin_memory=device.type=="cuda")
    val_dl=DataLoader(val_set,batch_size=cfg["batch_size"],shuffle=False,num_workers=0)
    model=UNet(cfg["input_channels"],cfg["base_channels"]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    use_amp=bool(cfg.get("amp_cuda",False)) and device.type=="cuda"
    scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    start_epoch=0
    best=-1.0

    if args.resume:
        saved=torch.load(args.resume,map_location="cpu",weights_only=True)
        old=saved.get("config",{})
        for key in ("input_channels","base_channels","image_size"):
            if key in old and old[key]!=cfg[key]:
                raise ValueError(f"Checkpoint incompatible: {key}")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        move_optimizer_states(optimizer,device)
        if use_amp and saved.get("scaler_state_dict"):
            scaler.load_state_dict(saved["scaler_state_dict"])
        start_epoch=int(saved["epoch"])+1
        best=float(saved.get("best_dice",-1))

    log=run_dir/"metrics.csv"
    append=bool(args.resume and log.exists())
    with log.open("a" if append else "w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        if not append: writer.writerow(["epoch","train_loss","val_loss","val_dice","val_iou","device"])
        for epoch in range(start_epoch,int(cfg["epochs"])):
            model.train()
            total=0.0
            count=0
            for images,masks in tqdm(train_dl,desc=f"Epoch {epoch+1}/{cfg['epochs']}"):
                images,masks=images.to(device),masks.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda",enabled=use_amp):
                    logits=model(images)
                    loss=bce_dice_loss(logits,masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total+=loss.item()*len(images)
                count+=len(images)
            vl,vd,vi=evaluate(model,val_dl,device)
            improved=vd>best
            if improved: best=vd
            state={"epoch":epoch,"model_state_dict":model.state_dict(),
                   "optimizer_state_dict":optimizer.state_dict(),
                   "scaler_state_dict":scaler.state_dict(),
                   "config":cfg,"best_dice":best,"torch_version":str(torch.__version__)}
            atomic_torch_save(state,run_dir/"latest.pt")
            if improved: atomic_torch_save(state,run_dir/"best.pt")
            writer.writerow([epoch+1,total/count,vl,vd,vi,device.type])
            f.flush()
            print(f"Epoch {epoch+1}: train_loss={total/count:.4f}; Dice={vd:.4f}; IoU={vi:.4f}")
    print("Saved outputs under:",run_dir)

if __name__=="__main__":
    main()
