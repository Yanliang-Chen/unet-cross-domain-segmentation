"""Benchmark scenarios and their 256-pixel image caches with native-resolution masks."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from datasets.files import index_images
from datasets.polypgen import center_pairs

ROOT = Path("data")
CACHE = Path("runs/benchmark/cache")
SCENARIOS = ("source_val", "clinicdb", "colondb", "etis", "polypgen", "far")


def _pairs(directory):
    images = index_images(Path(directory)/"images")
    masks = {p.stem: p for p in index_images(Path(directory)/"masks")}
    if {p.stem for p in images} != set(masks):
        raise ValueError(f"Image/mask mismatch in {directory}")
    return [(p, masks[p.stem]) for p in images]


def _row(image, mask, kind, group, role):
    return {"image": str(image), "mask": str(mask) if mask else "", "kind": kind,
            "group": group, "role": role}


def scenario_rows(name):
    """Rows with role 'val' (validator input), 'test' (labelled) or 'both' (transductive)."""
    if name == "source_val":
        return [_row(i, m, "polyp", "", "test") for i, m in _pairs(ROOT/"polyp_png/source_val")]
    if name == "clinicdb":
        with (ROOT/"polyp_png/split_manifest.csv").open(newline="", encoding="utf-8") as stream:
            groups = {r["sample_id"]: r["group_id"] for r in csv.DictReader(stream)}
        rows = [_row(p, None, "unknown", groups[p.stem], "val")
                for p in index_images(ROOT/"polyp_png/target_select/images")]
        rows += [_row(i, m, "polyp", groups[i.stem], "test") for i, m in _pairs(ROOT/"polyp_png/target_test")]
        if sum(r["role"] == "val" for r in rows) != 185 or len(rows) != 612:
            raise ValueError("ClinicDB split changed")
        return rows
    if name in ("colondb", "etis"):
        folder = {"colondb": "CVC-ColonDB", "etis": "ETIS-LaribPolypDB"}[name]
        rows = [_row(i, m, "polyp", "", "both") for i, m in _pairs(ROOT/"TestDataset/TestDataset"/folder)]
        if len(rows) != {"colondb": 380, "etis": 196}[name]:
            raise ValueError(f"{name} count changed")
        return rows
    if name == "polypgen":
        rows = [_row(i, m, "polyp", center, "both")
                for center, pairs in center_pairs().items() for i, m in pairs]
        split = json.loads(Path("runs/real/negative_context/split.json").read_text(encoding="utf-8"))
        rows += [_row(r["image"], None, "normal", r["analysis_group"], "both") for r in split["roles"]["holdout"]]
        if len(rows) != 1537 + 1755:
            raise ValueError("PolypGen count changed")
        return rows
    if name == "far":
        freeze = json.loads(Path("runs/real/far_confirmatory/freeze.json").read_text(encoding="utf-8"))
        mapping = freeze["patient_group_mapping"]
        rows = []
        for patient in sorted(mapping):
            folder = ROOT/"FAR-POLYP-SEG/Dataset"/patient
            for kind, sub in (("polyp", "Polyp"), ("normal", "Healthy")):
                if not (folder/sub).is_dir() or not any((folder/sub).iterdir()):
                    continue
                for image in index_images(folder/sub):
                    mask = folder/"BinaryMask"/image.name if kind == "polyp" else None
                    if mask is not None and not mask.is_file():
                        raise FileNotFoundError(mask)
                    rows.append(_row(image, mask, kind, mapping[patient], "both"))
        if (sum(r["kind"] == "polyp" for r in rows), sum(r["kind"] == "normal" for r in rows)) != (432, 7749):
            raise ValueError("FAR count changed")
        return rows
    raise ValueError(f"Unknown scenario {name}")


def build_cache(name, size=256):
    """Stores uint8 256 images, native boolean masks and native sizes once per scenario."""
    path = CACHE/f"{name}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    rows = scenario_rows(name)
    images = torch.empty((len(rows), 3, size, size), dtype=torch.uint8)
    masks, sizes = [], []
    for i, row in enumerate(rows):
        with Image.open(row["image"]) as im:
            rgb = im.convert("RGB")
            sizes.append((rgb.height, rgb.width))
            images[i] = torch.from_numpy(np.asarray(
                rgb.resize((size, size), Image.Resampling.BILINEAR)).transpose(2, 0, 1).copy())
        if row["mask"]:
            with Image.open(row["mask"]) as im:
                mask = torch.from_numpy(np.asarray(im.convert("L")) > 127)
            if tuple(mask.shape) != sizes[-1]:
                raise ValueError(f"Mask size differs: {row['mask']}")
            masks.append(mask)
        else:
            masks.append(None)
    payload = {"rows": rows, "images": images, "masks": masks, "sizes": sizes}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


if __name__ == "__main__":
    for scenario in SCENARIOS:
        cache = build_cache(scenario)
        kinds = {k: sum(r["kind"] == k for r in cache["rows"]) for k in ("polyp", "normal", "unknown")}
        print(scenario, len(cache["rows"]), kinds, flush=True)
