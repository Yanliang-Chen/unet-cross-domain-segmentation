"""Stage paired datasets without exposing target selection masks to the selector."""
import argparse
import csv
import hashlib
import random
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from PIL import Image

from datasets.files import index_images


def pairs(root):
    root = Path(root)
    if (root / "images").is_dir():
        image_dir, mask_dir = root / "images", root / "masks"
    elif (root / "Original").is_dir():
        image_dir, mask_dir = root / "Original", root / "Ground Truth"
    else:
        raise FileNotFoundError(f"Expected images/masks or Original/Ground Truth in {root}")
    images = index_images(image_dir)
    masks = {p.stem: p for p in index_images(mask_dir)}
    if {p.stem for p in images} != set(masks):
        raise ValueError(f"Image/mask stems do not match in {root}")
    return [(p, masks[p.stem]) for p in images]


def split_items(items, holdout_fraction, seed, groups=None):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if len(items) < 2:
        raise ValueError("At least two samples are needed for a split")
    rng = random.Random(seed)
    if groups is None:
        shuffled = list(items)
        rng.shuffle(shuffled)
        count = max(1, min(len(items) - 1, round(len(items) * holdout_fraction)))
        return shuffled[count:], shuffled[:count]
    missing = [image.stem for image, _ in items if image.stem not in groups]
    if missing:
        raise ValueError(f"Missing group IDs for {missing[:5]}")
    bucket = defaultdict(list)
    for item in items:
        bucket[groups[item[0].stem]].append(item)
    group_ids = sorted(bucket)
    if len(group_ids) < 2:
        raise ValueError("Group split needs at least two distinct group IDs")
    rng.shuffle(group_ids)
    count = max(1, min(len(group_ids) - 1, round(len(group_ids) * holdout_fraction)))
    holdout_ids = set(group_ids[:count])
    return (
        [item for item in items if groups[item[0].stem] not in holdout_ids],
        [item for item in items if groups[item[0].stem] in holdout_ids],
    )


def read_groups(path):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if {"sample_id", "group_id"} <= fields:
            sample_key, group_key = "sample_id", "group_id"
        elif {"frame_id", "sequence_id"} <= fields:
            sample_key, group_key = "frame_id", "sequence_id"
        else:
            raise ValueError("CSV needs sample_id,group_id or frame_id,sequence_id")
        groups = {}
        for row in reader:
            key, value = row[sample_key].strip(), row[group_key].strip()
            if not key or not value or key in groups:
                raise ValueError("Empty or duplicate sample_id/group_id")
            groups[key] = value
    return groups


def validate_pairs(items):
    """Reject corrupt or mismatched polyp image/mask pairs before staging."""
    for image, mask in items:
        with Image.open(image) as opened:
            opened.load()
            image_size = opened.size
        with Image.open(mask) as opened:
            opened.load()
            if opened.size != image_size:
                raise ValueError(f"Image/mask dimensions differ: {image}, {mask}")
            foreground = opened.convert("L").point(lambda value: 255 if value > 127 else 0)
            if foreground.getbbox() is None:
                raise ValueError(f"Empty polyp mask after thresholding: {mask}")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes_by_split(splits):
    """Catch exact duplicate images crossing train/validation or selection/test."""
    seen = {}
    hashes = {}
    for split, items in splits:
        for image, _ in items:
            digest = file_hash(image)
            previous = seen.get(digest)
            if previous is not None and previous != split:
                raise ValueError(
                    f"Duplicate image crosses splits: {image} in {split}; also in {previous}"
                )
            seen[digest] = split
            hashes[(split, image.stem)] = digest
    return hashes


def stage(items, destination, include_masks):
    for image, mask in items:
        images = destination / "images"
        images.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, images / image.name)
        if include_masks:
            masks = destination / "masks"
            masks.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mask, masks / mask.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Unzipped Kvasir-SEG directory containing images/ and masks/")
    parser.add_argument("--target", required=True,
                        help="Unzipped CVC-ClinicDB directory containing images/ and masks/")
    parser.add_argument("--output", default="data/polyp")
    parser.add_argument("--target-groups-csv",
                        help="Optional CSV sample_id,group_id; group target frames by patient/video")
    parser.add_argument("--source-groups-csv",
                        help="Optional CSV sample_id,group_id; group source images by patient")
    parser.add_argument("--require-target-groups", action="store_true",
                        help="Refuse exploratory target frame-level splitting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-val-fraction", type=float, default=0.2)
    parser.add_argument("--target-select-fraction", type=float, default=0.3)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    for raw in (args.source, args.target):
        raw = Path(raw).resolve()
        if raw == output or output in raw.parents or raw in output.parents:
            raise ValueError("Keep raw dataset directories separate from --output")
    if output.exists() and any(output.iterdir()):
        parser.error(f"{output} is not empty; choose a fresh output directory")

    source_pairs = pairs(args.source)
    target_pairs = pairs(args.target)
    validate_pairs(source_pairs)
    validate_pairs(target_pairs)
    source_groups = read_groups(args.source_groups_csv)
    target_groups = read_groups(args.target_groups_csv)
    if args.require_target_groups and target_groups is None:
        parser.error("--require-target-groups needs --target-groups-csv")
    source_train, source_val = split_items(
        source_pairs, args.source_val_fraction, args.seed, source_groups
    )
    target_test, target_select = split_items(
        target_pairs, args.target_select_fraction, args.seed, target_groups
    )
    splits = (
        ("source_train", source_train), ("source_val", source_val),
        ("target_select", target_select), ("target_test", target_test),
    )
    hashes = image_hashes_by_split(splits)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-staging-", dir=output.parent))
    try:
        for split, samples in splits:
            stage(samples, temporary / split, include_masks=split != "target_select")
        with (temporary / "split_manifest.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["split", "sample_id", "group_id", "image_sha256"])
            for split, samples in splits:
                groups = source_groups if split.startswith("source") else target_groups
                for image, _ in samples:
                    writer.writerow([
                        split, image.stem, groups[image.stem] if groups else "",
                        hashes[(split, image.stem)],
                    ])
        if output.exists():
            output.rmdir()  # The preflight check established that it was empty.
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"Source: {len(source_train)} train, {len(source_val)} val")
    print(f"Target: {len(target_select)} unlabeled selection, {len(target_test)} test")
    if target_groups is None:
        print("WARNING: target frames split randomly; video/patient overlap can bias test results.")
        print("Supply --target-groups-csv if original video/patient identifiers are available.")
    else:
        print("Target split performed by group IDs; verify IDs reflect true patient/video boundaries.")


if __name__ == "__main__":
    main()
