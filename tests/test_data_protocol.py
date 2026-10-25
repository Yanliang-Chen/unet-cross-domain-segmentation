"""Focused checks for data split isolation; runs without PyTorch."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.prepare_polyp_data import (
    image_hashes_by_split, pairs, read_groups, split_items, stage, validate_pairs,
)


class DataProtocolTests(unittest.TestCase):
    def test_group_split_and_no_target_selection_masks(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            raw = root / "raw"
            (raw / "Original").mkdir(parents=True)
            (raw / "Ground Truth").mkdir()
            for i in range(6):
                name = f"{i}.png"
                Image.new("RGB", (8, 8), color="red").save(raw / "Original" / name)
                Image.new("L", (8, 8), color=255).save(raw / "Ground Truth" / name)
            mapping = root / "metadata.csv"
            with mapping.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["frame_id", "sequence_id"])
                writer.writerows((str(i), f"video_{i // 2}") for i in range(6))
            groups = read_groups(mapping)
            train, selection = split_items(pairs(raw), 0.33, seed=42, groups=groups)
            self.assertFalse(
                {groups[image.stem] for image, _ in train}
                & {groups[image.stem] for image, _ in selection}
            )
            stage(selection, root / "target_select", include_masks=False)
            self.assertEqual(len(list((root / "target_select" / "images").glob("*.png"))),
                             len(selection))
            self.assertFalse((root / "target_select" / "masks").exists())

    def test_missing_group_id_fails(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            raw = Path(temporary)
            (raw / "images").mkdir()
            (raw / "masks").mkdir()
            for stem in ("one", "two"):
                Image.new("RGB", (4, 4)).save(raw / "images" / f"{stem}.png")
                Image.new("L", (4, 4)).save(raw / "masks" / f"{stem}.png")
            with self.assertRaisesRegex(ValueError, "Missing group IDs"):
                split_items(pairs(raw), 0.5, 42, {"one": "a"})

    def test_grouped_cli_stages_verified_pairs_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source, target, output = root / "source", root / "target", root / "prepared"
            for domain, folder in (("s", source), ("t", target)):
                (folder / "images").mkdir(parents=True)
                (folder / "masks").mkdir()
                for i in range(4):
                    stem = f"{domain}{i}"
                    Image.new(
                        "RGB", (12, 12), (i * 40, 30 if domain == "s" else 180, 70)
                    ).save(
                        folder / "images" / f"{stem}.png"
                    )
                    mask = Image.new("L", (12, 12))
                    ImageDraw.Draw(mask).rectangle((2, 2, 6, 6), fill=255)
                    mask.save(folder / "masks" / f"{stem}.png")
                with (folder / "groups.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["sample_id", "group_id"])
                    writer.writerows((f"{domain}{i}", f"{domain}_group_{i // 2}") for i in range(4))
            missing_groups = subprocess.run([
                sys.executable, "-m", "scripts.prepare_polyp_data",
                "--source", str(source), "--target", str(target),
                "--require-target-groups", "--output", str(root / "blocked"),
            ], capture_output=True, text=True, check=False)
            self.assertNotEqual(missing_groups.returncode, 0)
            self.assertIn("--target-groups-csv", missing_groups.stderr)
            self.assertFalse((root / "blocked").exists())
            result = subprocess.run([
                sys.executable, "-m", "scripts.prepare_polyp_data",
                "--source", str(source), "--target", str(target),
                "--source-groups-csv", str(source / "groups.csv"),
                "--target-groups-csv", str(target / "groups.csv"),
                "--require-target-groups", "--output", str(output),
            ], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((output / "target_select" / "masks").exists())
            with (output / "split_manifest.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            for prefix in ("source", "target"):
                split_a = "source_train" if prefix == "source" else "target_select"
                split_b = "source_val" if prefix == "source" else "target_test"
                groups_a = {row["group_id"] for row in rows if row["split"] == split_a}
                groups_b = {row["group_id"] for row in rows if row["split"] == split_b}
                self.assertFalse(groups_a & groups_b)
            self.assertTrue(all(len(row["image_sha256"]) == 64 for row in rows))

    def test_rejects_empty_mask_and_cross_split_duplicate(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            image = root / "one.png"
            duplicate = root / "two.png"
            mask = root / "mask.png"
            Image.new("RGB", (8, 8), "red").save(image)
            duplicate.write_bytes(image.read_bytes())
            Image.new("L", (8, 8)).save(mask)
            with self.assertRaisesRegex(ValueError, "Empty polyp mask"):
                validate_pairs([(image, mask)])
            with self.assertRaisesRegex(ValueError, "Duplicate image crosses splits"):
                image_hashes_by_split((
                    ("source_train", [(image, mask)]),
                    ("source_val", [(duplicate, mask)]),
                ))


if __name__ == "__main__":
    unittest.main()
