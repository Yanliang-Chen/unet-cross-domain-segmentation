"""Exploratory native-mask evaluation of CMR selection and two-validator ensemble."""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from datasets.segmentation import SegmentationDataset, load_image
from scripts.check_protocol import sha256
from scripts.evaluate_multiscale import checkpoint_model
from utils import choose_device, per_image_metrics


def domain_pairs(protocol):
    domains = {
        "CVC-ClinicDB": SegmentationDataset("data/polyp_png", "target_test", 256, 3).pairs,
        "CVC-ColonDB": SegmentationDataset("data/TestDataset/TestDataset", "CVC-ColonDB", 256, 3).pairs,
        "ETIS-LaribPolypDB": SegmentationDataset("data/TestDataset/TestDataset", "ETIS-LaribPolypDB", 256, 3).pairs,
    }
    if {name: len(pairs) for name, pairs in domains.items()} != protocol["evaluation_domains"]:
        raise ValueError("Evaluation dataset counts differ from the protocol")
    return domains


def summarize(rows, cutoff):
    small = [row for row in rows if row["ground_truth_fraction"] < cutoff]
    return {"n_images": len(rows), "n_small": len(small),
            **{f"mean_{key}": statistics.mean(row[key] for row in rows)
               for key in ("dice", "iou", "precision", "recall")},
            "small_mean_dice": statistics.mean(row["dice"] for row in small),
            "empty_prediction_fraction": statistics.mean(
                row["foreground_fraction"] == 0 for row in rows)}


@torch.inference_mode()
def evaluate_pair(source, cmr, pairs, device, threshold, weights):
    outputs = {"source": [], "cmr": [], "ensemble": []}
    for image_path, mask_path in pairs:
        image = load_image(image_path, 256, 3)[None].to(device)
        with Image.open(mask_path) as mask:
            target = torch.from_numpy((np.asarray(mask.convert("L")) > 127).copy())
        target = target[None, None].to(device=device, dtype=torch.float32)
        source_probability = F.interpolate(source(image).sigmoid(), size=target.shape[-2:],
                                           mode="bilinear", align_corners=False)
        if cmr is source:
            cmr_probability = source_probability
        else:
            cmr_probability = F.interpolate(cmr(image).sigmoid(), size=target.shape[-2:],
                                            mode="bilinear", align_corners=False)
        probabilities = {"source": source_probability, "cmr": cmr_probability,
                         "ensemble": weights[0] * source_probability + weights[1] * cmr_probability}
        for name, probability in probabilities.items():
            logits = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
            values = per_image_metrics(logits, target, threshold)[0]
            outputs[name].append({"sample_id": image_path.stem,
                                  "ground_truth_fraction": float(target.mean()), **values})
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="paper_selection_protocol.yaml")
    parser.add_argument("--selection", default="runs/real/paper_selection/cmr_selection.json")
    parser.add_argument("--output", default="runs/real/paper_selection/evaluation.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    protocol = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    if (protocol["ensemble_members"] != ["source", "cmr"]
            or len(protocol["ensemble_weights"]) != 2
            or abs(sum(protocol["ensemble_weights"]) - 1) > 1e-9):
        raise ValueError("Unsupported ensemble protocol")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if sha256("protocol.yaml") != protocol["original_protocol_sha256"]:
        raise ValueError("Original protocol changed")
    if selection["protocol_sha256"] != sha256(args.protocol):
        raise ValueError("CMR selection was not frozen under this protocol")
    expected_runs = {f"{augmentation}_seed{seed}" for augmentation in protocol["augmentations"]
                     for seed in protocol["seeds"]}
    if {run["run"] for run in selection["runs"]} != expected_runs or len(selection["runs"]) != len(expected_runs):
        raise ValueError("Incomplete selection")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    # Validate the selected model identities before opening evaluation masks.
    for run in selection["runs"]:
        for choice in run["methods"].values():
            if sha256(choice["checkpoint"]) != choice["checkpoint_sha256"]:
                raise ValueError(f"Changed checkpoint: {choice['checkpoint']}")
    domains = domain_pairs(protocol)
    device = choose_device(args.device)
    results = []
    for run in selection["runs"]:
        models = {}
        for name, choice in run["methods"].items():
            path = choice["checkpoint"]
            if path not in models:
                model, size = checkpoint_model(path, choice["checkpoint_sha256"], run["seed"],
                                               run["augmentation"], device)
                if size != 256:
                    raise ValueError("Unexpected input image size")
                models[path] = model.eval()
        source = models[run["methods"]["source"]["checkpoint"]]
        cmr = models[run["methods"]["cmr"]["checkpoint"]]
        for domain, pairs in domains.items():
            per_image = evaluate_pair(source, cmr, pairs, device,
                                      float(protocol["prediction_threshold"]),
                                      protocol["ensemble_weights"])
            summaries = {name: summarize(rows, float(protocol["small_polyp_area_cutoff"]))
                         for name, rows in per_image.items()}
            results.append({"dataset": domain, "run": run["run"], "seed": run["seed"],
                            "augmentation": run["augmentation"], "methods": summaries,
                            "per_image": per_image})
            print(run["run"], domain,
                  *(f"{name}={summaries[name]['mean_dice']:.4f}"
                    for name in ("source", "cmr", "ensemble")), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": protocol["evaluation_status"],
                                  "protocol_sha256": sha256(args.protocol),
                                  "selection_sha256": sha256(args.selection),
                                  "results": results}, indent=2) + "\n", encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
