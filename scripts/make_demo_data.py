"""Generate synthetic shapes for installation testing, NOT medical research."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFilter

def main():
    rng=np.random.default_rng(42)
    root=Path("data/demo")
    for split,n in (("train",24),("val",8),("test",8)):
        for part in ("images","masks"):
            (root/split/part).mkdir(parents=True,exist_ok=True)
        for i in range(n):
            size=128
            base=np.clip(rng.normal(50,20,(size,size,3)),0,255).astype(np.uint8)
            x,y=[int(v) for v in rng.integers(35,95,size=2)]
            r=int(rng.integers(12,30))
            mask=Image.new("L",(size,size),0)
            ImageDraw.Draw(mask).ellipse((x-r,y-r,x+r,y+r),fill=255)
            region=np.asarray(mask).astype(bool)
            base[region,0]=np.clip(base[region,0].astype(int)+145,0,255)
            base[region,1]=np.clip(base[region,1].astype(int)+65,0,255)
            Image.fromarray(base,"RGB").filter(ImageFilter.GaussianBlur(0.4)).save(root/split/"images"/f"case_{i:03}.png")
            mask.save(root/split/"masks"/f"case_{i:03}.png")
        print(f"Generated {split}: {n} pairs")
    print("Synthetic smoke-test data only.")

if __name__=="__main__":
    main()
