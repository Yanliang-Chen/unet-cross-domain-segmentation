"""Generate synthetic shapes for a workflow smoke test, never for medical research."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def main():
    rng = np.random.default_rng(42)
    root = Path("data/demo")
    for split, n in (("source_train", 24), ("source_val", 8),
                     ("target_select", 8), ("target_test", 8)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        if split != "target_select":
            (root / split / "masks").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            size = 64
            baseline = 55 if split.startswith("source") else 75
            base = np.clip(rng.normal(baseline, 18, (size, size, 3)), 0, 255).astype(np.uint8)
            x, y = [int(v) for v in rng.integers(20, 44, size=2)]
            radius = int(rng.integers(7, 17))
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=255
            )
            region = np.asarray(mask).astype(bool)
            base[region, 0] = np.clip(base[region, 0].astype(int) + 140, 0, 255)
            base[region, 1] = np.clip(base[region, 1].astype(int) + 55, 0, 255)
            Image.fromarray(base, "RGB").filter(ImageFilter.GaussianBlur(0.4)).save(
                root / split / "images" / f"case_{i:03}.png"
            )
            if split != "target_select":
                mask.save(root / split / "masks" / f"case_{i:03}.png")
        print(f"{split}: {n} cases")
    print("Synthetic smoke-test data only. Never report its Dice as a medical result.")


if __name__ == "__main__":
    main()