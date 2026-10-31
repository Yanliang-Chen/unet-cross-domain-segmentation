"""Checks for the frozen experiment's leakage and metric boundaries."""
import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
import yaml
from PIL import Image

from scripts.check_protocol import check_protocol, sha256
from train import augment
from utils import per_image_metrics


class ExperimentTests(unittest.TestCase):

    def test_protocol_lock_detects_changed_staged_image(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            data = root / "data"
            rows = []
            for index, split in enumerate(("source_train", "source_val",
                                           "target_select", "target_test")):
                image_dir = data / split / "images"
                image_dir.mkdir(parents=True)
                image = image_dir / f"case_{index}.png"
                Image.new("RGB", (4, 4), (index * 30, 0, 0)).save(image)
                if split != "target_select":
                    mask_dir = data / split / "masks"
                    mask_dir.mkdir()
                    Image.new("L", (4, 4), 255).save(mask_dir / image.name)
                rows.append([split, image.stem, split if split.startswith("target") else "",
                             sha256(image)])
            manifest = data / "split_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["split", "sample_id", "group_id", "image_sha256"])
                writer.writerows(rows)
            protocol = root / "protocol.yaml"
            protocol.write_text(yaml.safe_dump({
                "data_dir": str(data), "split_manifest_sha256": sha256(manifest),
                "expected_split_counts": {row[0]: 1 for row in rows},
                "expected_target_groups": {"target_select": 1, "target_test": 1},
            }), encoding="utf-8")
            self.assertEqual(check_protocol(protocol)["manifest_sha256"], sha256(manifest))
            Image.new("RGB", (4, 4), "red").save(data / "target_test/images/case_3.png")
            with self.assertRaisesRegex(ValueError, "differs from the locked manifest"):
                check_protocol(protocol)

    def test_a0_flips_image_and_mask_together_without_color_change(self):
        image = torch.arange(4, dtype=torch.float32).view(1, 1, 2, 2) / 4
        mask = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
        with patch("train.torch.rand", side_effect=[torch.tensor(0.1), torch.tensor(0.9)]):
            changed_image, changed_mask = augment(image, mask, "a0")
        torch.testing.assert_close(changed_image, image.flip(-1))
        torch.testing.assert_close(changed_mask, mask.flip(-1))

    def test_a1_color_change_never_changes_mask(self):
        image = torch.full((1, 3, 2, 2), 0.5)
        mask = torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])
        draws = [torch.tensor(0.9), torch.tensor(0.9), torch.tensor(0.1),
                 torch.tensor(1.0), torch.tensor(0.5), torch.ones(3)]
        with patch("train.torch.rand", side_effect=draws):
            changed_image, changed_mask = augment(image, mask, "a1")
        torch.testing.assert_close(changed_mask, mask)
        self.assertFalse(torch.equal(changed_image, image))

    def test_a0_a1_share_geometry_random_stream(self):
        image = torch.rand(1, 3, 4, 4)
        mask = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
        a0_geometry = torch.Generator().manual_seed(2042)
        a1_geometry = torch.Generator().manual_seed(2042)
        a1_color = torch.Generator().manual_seed(3042)
        for _ in range(5):
            _, a0_mask = augment(image, mask, "a0", a0_geometry)
            _, a1_mask = augment(image, mask, "a1", a1_geometry, a1_color)
            torch.testing.assert_close(a0_mask, a1_mask)
        torch.testing.assert_close(a0_geometry.get_state(), a1_geometry.get_state())


    def test_per_image_metric_boundaries(self):
        logits = torch.tensor([[[[-20.0, 20.0]]], [[[-20.0, -20.0]]],
                               [[[-20.0, -20.0]]]])
        masks = torch.tensor([[[[0.0, 1.0]]], [[[0.0, 1.0]]],
                              [[[0.0, 0.0]]]])
        rows = per_image_metrics(logits, masks)
        for metric in ("dice", "iou", "precision", "recall"):
            self.assertEqual([row[metric] for row in rows], [1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
