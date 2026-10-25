"""Paired image/mask data and strictly image-only target selection data."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from datasets.files import index_images


def load_image(path, size, channels):
    mode = "RGB" if channels == 3 else "L"
    with Image.open(path) as image:
        array = np.asarray(
            image.convert(mode).resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    if channels == 1:
        array = array[..., None]
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class ImagesOnlyDataset(Dataset):
    """Does not open or index mask paths. Safe for target-domain model selection."""

    def __init__(self, image_dir, image_size, input_channels=3):
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        self.paths = index_images(image_dir)
        self.size = int(image_size)
        self.channels = input_channels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return load_image(self.paths[index], self.size, self.channels)


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split, image_size, input_channels=3):
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3")
        root = Path(data_dir) / split
        images = index_images(root / "images")
        masks = index_images(root / "masks")
        by_stem = {p.stem: p for p in masks}
        image_stems = {p.stem for p in images}
        if image_stems != set(by_stem):
            raise ValueError(
                f"Image/mask mismatch in {root}: "
                f"missing masks={sorted(image_stems - set(by_stem))[:5]}, "
                f"extra masks={sorted(set(by_stem) - image_stems)[:5]}"
            )
        self.pairs = [(p, by_stem[p.stem]) for p in images]
        self.size = int(image_size)
        self.channels = input_channels

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        x = load_image(image_path, self.size, self.channels)
        with Image.open(mask_path) as mask:
            # Original Kvasir masks are JPEG; compression creates low nonzero values.
            y = (np.asarray(
                mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
            ) > 127).astype(np.float32)
        return x, torch.from_numpy(y[None, ...].copy())