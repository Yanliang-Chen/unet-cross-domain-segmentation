"""Filesystem-only utilities shared with data preparation."""
from pathlib import Path

EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def index_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXT)
    stems = [p.stem for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f"Duplicate image stems in {directory}")
    if not paths:
        raise ValueError(f"No images in {directory}")
    return paths