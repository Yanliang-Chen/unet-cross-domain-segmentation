"""8-bit image and binary mask pairs with the same filename stem."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

EXT={".png",".jpg",".jpeg",".tif",".tiff",".bmp"}

class SegmentationDataset(Dataset):
    def __init__(self,data_dir,split,image_size,input_channels=3):
        if input_channels not in (1,3):
            raise ValueError("input_channels must be 1 or 3")
        root=Path(data_dir)/split
        image_dir,mask_dir=root/"images",root/"masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing {image_dir} or {mask_dir}")
        masks={p.stem:p for p in mask_dir.iterdir() if p.suffix.lower() in EXT}
        images=sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXT)
        self.pairs=[]
        for p in images:
            if p.stem not in masks:
                raise ValueError(f"Missing mask for {p}")
            self.pairs.append((p,masks[p.stem]))
        if not self.pairs: raise ValueError(f"No image/mask pairs under {root}")
        self.size=int(image_size)
        self.channels=input_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self,index):
        image_path,mask_path=self.pairs[index]
        mode="RGB" if self.channels==3 else "L"
        with Image.open(image_path) as im:
            x=np.asarray(im.convert(mode).resize((self.size,self.size),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        if self.channels==1: x=x[...,None]
        with Image.open(mask_path) as im:
            y=(np.asarray(im.convert("L").resize((self.size,self.size),Image.Resampling.NEAREST))>0).astype(np.float32)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(y[None,...].copy())
