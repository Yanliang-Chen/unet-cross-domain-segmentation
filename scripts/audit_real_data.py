"""Read-only audit of raw and staged Kvasir-SEG/CVC-ClinicDB data.

This script is self-contained so it can be downloaded into an ignored data/
directory and run while the working tree has unresolved Git conflicts.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SPLITS = ("source_train", "source_val", "target_select", "target_test")
# Frame ranges transcribed from the sequence table in the Tianchi CVC-ClinicDB
# dataset documentation (https://tianchi.aliyun.com/dataset/93690).
# This checks the supplied metadata against that documented table; the Tianchi
# page itself is a secondary distribution of the original CVC-ClinicDB data.
DOCUMENTED_SEQUENCE_ENDS = (
    25, 50, 67, 78, 103, 126, 151, 177, 199, 205,
    227, 252, 277, 297, 317, 342, 363, 383, 408, 428,
    447, 466, 478, 503, 528, 546, 571, 591, 612,
)


def documented_groups():
    groups = {}
    start = 1
    for sequence_id, end in enumerate(DOCUMENTED_SEQUENCE_ENDS, 1):
        groups.update((str(frame), str(sequence_id)) for frame in range(start, end + 1))
        start = end + 1
    return groups


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_files(directory, errors):
    if not directory.is_dir():
        errors.append(f"Missing directory: {directory}")
        return {}
    files = sorted(
        (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name,
    )
    indexed = {}
    for path in files:
        if path.stem in indexed:
            errors.append(f"Duplicate filename stem in {directory}: {path.stem}")
        indexed[path.stem] = path
    if not files:
        errors.append(f"No image files in {directory}")
    return indexed


def raw_pairs(root, errors):
    if (root / "images").is_dir():
        image_dir, mask_dir = root / "images", root / "masks"
    else:
        image_dir, mask_dir = root / "Original", root / "Ground Truth"
    images = index_files(image_dir, errors)
    masks = index_files(mask_dir, errors)
    if images.keys() != masks.keys():
        errors.append(
            f"Unpaired files in {root}: images only={sorted(images.keys() - masks.keys())[:5]}, "
            f"masks only={sorted(masks.keys() - images.keys())[:5]}"
        )
    return images, masks


def read_metadata(path, errors, target_format):
    groups = {}
    image_column = f"{target_format}_image_path"
    mask_column = f"{target_format}_mask_path"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not {"frame_id", "sequence_id", image_column, mask_column} <= fields:
            errors.append(f"Metadata lacks frame_id, sequence_id or {target_format.upper()} paths")
            return groups
        for row in reader:
            frame = row["frame_id"].strip()
            group = row["sequence_id"].strip()
            if not frame or not group or frame in groups:
                errors.append(f"Empty or repeated metadata frame/sequence: {frame!r}, {group!r}")
                continue
            groups[frame] = group
            for column, expected_folder in (
                (image_column, "Original"),
                (mask_column, "Ground Truth"),
            ):
                value = row[column].replace("\\", "/")
                parts = Path(value).parts
                if (
                    Path(value).stem != frame
                    or Path(value).suffix.lower() not in (
                        {".tif", ".tiff"} if target_format == "tif" else {".png"}
                    )
                    or expected_folder not in parts
                ):
                    errors.append(f"Metadata {column} does not match frame {frame}: {value}")
    return groups


def audit(source, target, metadata, prepared, archive=None, target_format="tif"):
    errors = []
    source_images, source_masks = raw_pairs(source, errors)
    target_images, target_masks = raw_pairs(target, errors)
    groups = read_metadata(metadata, errors, target_format)
    if len(source_images) != 1000 or len(source_masks) != 1000:
        errors.append(f"Expected 1000 source image/mask pairs, got {len(source_images)}/{len(source_masks)}")
    if len(target_images) != 612 or len(target_masks) != 612:
        errors.append(f"Expected 612 target image/mask pairs, got {len(target_images)}/{len(target_masks)}")
    if len(groups) != 612 or len(set(groups.values())) != 29:
        errors.append(f"Expected 612 metadata rows and 29 sequences, got {len(groups)}/{len(set(groups.values()))}")
    expected_groups = documented_groups()
    sequence_range_match = groups == expected_groups
    if not sequence_range_match:
        mismatches = [
            frame for frame in expected_groups if groups.get(frame) != expected_groups[frame]
        ]
        errors.append(
            "Metadata sequence IDs differ from the Tianchi documentation's "
            f"29 frame ranges; first mismatches={mismatches[:5]}"
        )
    if groups.keys() != target_images.keys():
        errors.append(
            f"Metadata frame IDs do not match target {target_format.upper()} filenames: "
            f"metadata only={sorted(groups.keys() - target_images.keys())[:5]}, "
            f"images only={sorted(target_images.keys() - groups.keys())[:5]}"
        )

    validated_pairs = 0
    for domain, images, masks in (
        ("source", source_images, source_masks),
        ("target", target_images, target_masks),
    ):
        for stem in sorted(images.keys() & masks.keys()):
            try:
                with Image.open(images[stem]) as image, Image.open(masks[stem]) as mask:
                    image.load()
                    mask.load()
                    if image.size != mask.size:
                        errors.append(f"Different image/mask dimensions: {domain}/{stem}")
                    if mask.convert("L").getextrema()[1] <= 127:
                        errors.append(f"Empty mask after >127 threshold: {domain}/{stem}")
                validated_pairs += 1
            except Exception as exc:
                errors.append(f"Unreadable pair {domain}/{stem}: {type(exc).__name__}: {exc}")

    manifest = prepared / "split_manifest.csv"
    if not manifest.is_file():
        errors.append(f"Missing split manifest: {manifest}")
        rows = []
        group_column = ""
    else:
        with manifest.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            group_column = "group_id" if "group_id" in columns else "target_group_id"
            if not {"split", "sample_id", group_column} <= columns:
                errors.append("Split manifest lacks split, sample_id or target group column")
            rows = list(reader)

    by_split = defaultdict(list)
    for row in rows:
        split = row.get("split", "")
        if split not in SPLITS:
            errors.append(f"Unknown split in manifest: {split}")
            continue
        by_split[split].append(row)
    if sum(len(by_split[s]) for s in ("source_train", "source_val")) != 1000:
        errors.append("Source split counts do not total 1000")
    if sum(len(by_split[s]) for s in ("target_select", "target_test")) != 612:
        errors.append("Target split counts do not total 612")

    seen_hashes = {}
    staged_ids = {"source": [], "target": []}
    split_groups = {}
    for split in SPLITS:
        is_target = split.startswith("target")
        raw_images = target_images if is_target else source_images
        raw_masks = target_masks if is_target else source_masks
        staged_images = index_files(prepared / split / "images", errors)
        mask_dir = prepared / split / "masks"
        if split == "target_select":
            if mask_dir.exists():
                errors.append("target_select contains a masks directory")
            staged_masks = {}
        else:
            staged_masks = index_files(mask_dir, errors)
        manifest_ids = [row.get("sample_id", "") for row in by_split[split]]
        if len(manifest_ids) != len(set(manifest_ids)) or set(manifest_ids) != staged_images.keys():
            errors.append(f"Manifest and staged image IDs differ in {split}")
        if split != "target_select" and staged_masks.keys() != staged_images.keys():
            errors.append(f"Staged images/masks differ in {split}")
        staged_ids["target" if is_target else "source"].extend(staged_images)
        split_groups[split] = set()
        for row in by_split[split]:
            stem = row.get("sample_id", "")
            if is_target:
                group = groups.get(stem)
                if not group or row.get(group_column, "") != group:
                    errors.append(f"Wrong or missing sequence ID in manifest: {split}/{stem}")
                else:
                    split_groups[split].add(group)
            if stem not in staged_images or stem not in raw_images:
                continue
            staged_hash = file_hash(staged_images[stem])
            if staged_hash != file_hash(raw_images[stem]):
                errors.append(f"Staged image differs from raw image: {split}/{stem}")
            previous = seen_hashes.get(staged_hash)
            if previous and previous != split:
                errors.append(f"Identical image in different splits: {previous} and {split}/{stem}")
            seen_hashes[staged_hash] = split
            if split != "target_select" and stem in staged_masks and stem in raw_masks:
                if file_hash(staged_masks[stem]) != file_hash(raw_masks[stem]):
                    errors.append(f"Staged mask differs from raw mask: {split}/{stem}")
    for domain, expected in (("source", source_images), ("target", target_images)):
        ids = staged_ids[domain]
        if len(ids) != len(set(ids)) or set(ids) != expected.keys():
            errors.append(f"Raw {domain} IDs are not partitioned exactly once")
    overlap = split_groups.get("target_select", set()) & split_groups.get("target_test", set())
    if overlap:
        errors.append(f"Target sequence IDs cross selection/test: {sorted(overlap)}")

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "raw_source_pairs": len(source_images),
        "raw_target_pairs": len(target_images),
        "target_format": target_format,
        "validated_pairs": validated_pairs,
        "metadata_rows": len(groups),
        "metadata_sequences": len(set(groups.values())),
        "sequence_mapping_reference": "https://tianchi.aliyun.com/dataset/93690",
        "sequence_mapping_matches_documentation": sequence_range_match,
        "metadata_sha256": file_hash(metadata),
        "archive_sha256": file_hash(archive) if archive else None,
        "split_counts": {split: len(by_split[split]) for split in SPLITS},
        "target_select_sequences": len(split_groups.get("target_select", set())),
        "target_test_sequences": len(split_groups.get("target_test", set())),
        "target_sequence_overlap": len(overlap),
        "errors": errors[:30],
        "error_count": len(errors),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional original CVC ZIP for provenance hash")
    parser.add_argument("--target-format", choices=("tif", "png"), default="tif")
    parser.add_argument("--report", type=Path, help="Optional JSON report under ignored data/")
    args = parser.parse_args()
    summary = audit(
        args.source, args.target, args.metadata, args.prepared,
        args.archive, args.target_format,
    )
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
