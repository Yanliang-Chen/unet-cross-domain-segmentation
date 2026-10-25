"""Focused checks for data split isolation; runs without PyTorch."""
import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_polyp_data import pairs, split_items, stage, read_groups


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


if __name__ == "__main__":
    unittest.main()