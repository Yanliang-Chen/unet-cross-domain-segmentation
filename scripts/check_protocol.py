"""Check the fixed real-data split before any training or target evaluation."""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from datasets.files import index_images


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_protocol(protocol_path, data_dir=None, require_lock=True):
    protocol_path = Path(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    root = Path(data_dir or protocol["data_dir"])
    manifest = root / "split_manifest.csv"
    digest = sha256(manifest)
    expected_digest = protocol.get("split_manifest_sha256")
    if require_lock and not expected_digest:
        raise ValueError(
            f"Set split_manifest_sha256 in {protocol_path} to {digest} and commit it "
            "before real training"
        )
    if expected_digest and digest != expected_digest:
        raise ValueError(f"Split manifest SHA-256 differs: actual={digest}, locked={expected_digest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "sample_id", "group_id", "image_sha256"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Manifest needs columns {sorted(required)}")
    counts = Counter(row["split"] for row in rows)
    expected_counts = protocol["expected_split_counts"]
    if dict(counts) != expected_counts:
        raise ValueError(f"Split counts differ: actual={dict(counts)}, expected={expected_counts}")
    seen = set()
    groups = defaultdict(set)
    for row in rows:
        split, sample_id = row["split"], row["sample_id"]
        if not sample_id or (split, sample_id) in seen:
            raise ValueError(f"Empty or duplicate sample ID: {split}/{sample_id}")
        seen.add((split, sample_id))
        if len(row["image_sha256"]) != 64:
            raise ValueError(f"Invalid image SHA-256 for {split}/{sample_id}")
        if split.startswith("target"):
            if not row["group_id"]:
                raise ValueError(f"Missing target group for {split}/{sample_id}")
            groups[split].add(row["group_id"])
    if groups["target_select"] & groups["target_test"]:
        raise ValueError("Target sequence IDs overlap between selection and test")
    for split, expected in protocol["expected_target_groups"].items():
        if len(groups[split]) != expected:
            raise ValueError(f"{split} group count is {len(groups[split])}, expected {expected}")
    if (root / "target_select" / "masks").exists():
        raise ValueError("target_select must not have a masks directory")
    for split, count in expected_counts.items():
        image_paths = index_images(root / split / "images")
        if len(image_paths) != count:
            raise ValueError(f"{split} has {len(image_paths)} staged images, expected {count}")
        if {p.stem for p in image_paths} != {
            row["sample_id"] for row in rows if row["split"] == split
        }:
            raise ValueError(f"{split} staged image IDs differ from manifest")
        expected_hashes = {row["sample_id"]: row["image_sha256"] for row in rows
                           if row["split"] == split}
        for image in image_paths:
            if sha256(image) != expected_hashes[image.stem]:
                raise ValueError(f"{split}/{image.name} differs from the locked manifest")
        if split != "target_select":
            mask_paths = index_images(root / split / "masks")
            if {p.stem for p in mask_paths} != {p.stem for p in image_paths}:
                raise ValueError(f"{split} staged image/mask IDs differ")
    return {"manifest_sha256": digest, "split_counts": dict(counts),
            "target_group_counts": {key: len(value) for key, value in groups.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="protocol.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--show-unlocked-hash", action="store_true",
                        help="Validate the staging and print its hash before locking")
    args = parser.parse_args()
    result = check_protocol(args.protocol, args.data_dir,
                            require_lock=not args.show_unlocked_hash)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
