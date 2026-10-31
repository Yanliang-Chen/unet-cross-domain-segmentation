"""PolypGen single-image pairing, independent of retired adaptation experiments."""
from pathlib import Path

DEFAULT_ROOT = Path("data/PolypGen2021_MultiCenterData_v3/PolypGen2021_MultiCenterData_v3")
CENTER_COUNTS = {"C1":256, "C2":301, "C3":457, "C4":227, "C5":208, "C6":88}
PAIRING_EXCEPTIONS = {"C3_EndoCV2021_00489]":"C3_EndoCV2021_00489"}


def center_pairs(root=DEFAULT_ROOT, counts=None, exceptions=None):
    root = Path(root)
    counts = CENTER_COUNTS if counts is None else counts
    exceptions = PAIRING_EXCEPTIONS if exceptions is None else exceptions
    domains = {}
    for center, count in counts.items():
        image_dir = root/f"data_{center}"/f"images_{center}"
        mask_dir = root/f"data_{center}"/f"masks_{center}"
        images, masks = sorted(image_dir.glob("*.jpg")), sorted(mask_dir.glob("*.jpg"))
        if len(images) != count or len(masks) != count:
            raise ValueError(f"Unexpected {center} image/mask count")
        if {exceptions.get(p.stem,p.stem) for p in images} != {p.stem.removesuffix("_mask") for p in masks}:
            raise ValueError(f"Image/mask mismatch in {center}")
        domains[center] = [(p,mask_dir/f"{exceptions.get(p.stem,p.stem)}_mask.jpg") for p in images]
    return domains
