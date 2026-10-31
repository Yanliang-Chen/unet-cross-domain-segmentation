"""Frozen feature construction for the retained relative-presence module.

No target labels select the head or threshold. Previously unlocked outcomes are
reused algebraically when a gate keeps the mask or replaces it by an empty mask.
"""
import argparse
import csv
from scripts.artifacts import read_json, write_json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import yaml

from datasets.segmentation import SegmentationDataset, load_image
from models.presence_gate import unet_features, pool_features
from scripts.check_protocol import sha256
from scripts.checkpoints import checkpoint_model
from datasets.polypgen import center_pairs
from utils import atomic_torch_save, choose_device

ROOT = Path("runs/real/presence_features_clean")
PROTOCOL = Path("presence_protocol.yaml")
EVALUATION = Path("presence_evaluation_protocol.yaml")


def setup():
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    frozen = read_json(ROOT/"frozen.json")
    for path, digest in frozen["identities"].items():
        if sha256(path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return p, frozen


def freeze():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    negatives = read_json(p["negative_split"])
    auxiliary = read_json(p["auxiliary_split"])
    adaptation = read_json(p["segmentation_manifest"])
    if adaptation["protocol_sha256"] != sha256("external_teacher_protocol.yaml"):
        raise ValueError("Original student identity changed")
    runs = sorted([r for r in adaptation["runs"] if r["arm"] == p["segmentation_arm"]], key=lambda r:r["run"])
    expected = {f"{a}_seed{s}" for a in p["augmentations"] for s in p["seeds"]}
    if len(runs) != 6 or {r["run"] for r in runs} != expected:
        raise ValueError("Incomplete six frozen segmentation students")
    source = []
    for role, split in (("original_positive", "source_train"), ("positive_calibration", "source_val")):
        source.extend({"image":str(image.resolve()), "sample_id":image.stem, "role":role,
                       "label":1, "group":"", "domain":"source_"+role}
                      for image, _ in SegmentationDataset(p["source_data"], split, 256, 3).pairs)
    source.extend({"image":r["image"], "sample_id":Path(r["image"]).stem,
                   "role":"auxiliary_positive", "label":1, "group":r["center"], "domain":"source_auxiliary_positive"}
                  for r in auxiliary["positive_pairs"])
    for role, old_role in (("negative", "train"), ("negative_validation", "diagnostic")):
        source.extend({"image":r["image"], "sample_id":Path(r["image"]).stem,
                       "role":role, "label":0, "group":r["group"],
                       "domain":"normal_diagnostic" if old_role == "diagnostic" else "source_negative"}
                      for r in negatives["roles"][old_role])
    counts = dict(Counter(r["role"] for r in source))
    if counts != p["source_counts"]:
        raise ValueError(f"Source roles changed: {counts}")
    with Path("data/polyp_png/split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        groups = {r["sample_id"]:r["group_id"] for r in csv.DictReader(stream) if r["split"] == "target_test"}
    c6 = center_pairs(counts={"C6":88})["C6"]
    evaluation = []
    for domain, pairs in (("CVC-ClinicDB", SegmentationDataset(p["source_data"], "target_test", 256, 3).pairs),
                          ("PolypGen-C6", c6)):
        evaluation.extend({"image":str(image.resolve()), "sample_id":image.stem, "domain":domain,
                           "group":groups[image.stem] if domain == "CVC-ClinicDB" else ""}
                          for image, _ in pairs)
    evaluation.extend({"image":r["image"], "sample_id":Path(r["image"]).stem, "domain":"normal_holdout",
                       "group":r.get("analysis_group", r["group"])} for r in negatives["roles"]["holdout"])
    if dict(Counter(r["domain"] for r in evaluation)) != {k:v for k,v in p["evaluation_counts"].items() if k != "normal_diagnostic"}:
        raise ValueError("Evaluation cohort changed")
    # Use prior data identities rather than introducing a new patient audit.
    for r in source+evaluation:
        r["image_sha256"] = sha256(r["image"])
    if {r["image_sha256"] for r in source} & {r["image_sha256"] for r in evaluation}:
        raise ValueError("Exact source/evaluation image overlap")
    paths = [PROTOCOL, EVALUATION, Path(__file__), Path("models/presence_gate.py"),
             Path(p["segmentation_manifest"]), Path(p["negative_split"]),
             Path(p["auxiliary_split"]), Path(p["frozen_comparator_csv"]), Path(p["sam_checkpoint"])]
    paths.extend(Path(r["best_checkpoint"]) for r in runs)
    ROOT.mkdir(parents=True)
    write_json(ROOT/"frozen.json", {"frozen_utc":datetime.now(timezone.utc).isoformat(),
        "scope":p["status"], "identities":{str(path):sha256(path) for path in paths},
        "source_rows":source, "evaluation_rows":evaluation, "runs":runs})
    print("Frozen", counts, "evaluation", len(evaluation), flush=True)


@torch.inference_mode()
def cache(which):
    from segment_anything import sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide
    p, frozen = setup()
    if which == "evaluation" and not Path(p["head_manifest"]).exists():
        raise ValueError("Fit and freeze all source-calibrated heads before evaluation")
    destination = ROOT/f"{which}_features.pt"
    if destination.exists():
        raise FileExistsError(destination)
    rows = frozen[which+"_rows"]
    device = choose_device("cuda")
    torch.set_num_threads(4)
    sam = sam_model_registry[p["sam_model"]](checkpoint=p["sam_checkpoint"]).to(device).eval()
    sam.requires_grad_(False)
    transform = ResizeLongestSide(sam.image_encoder.img_size)
    models = {}
    for r in frozen["runs"]:
        model, size = checkpoint_model(r["best_checkpoint"], r["best_checkpoint_sha256"],
                                       r["seed"], r["augmentation"], device)
        if size != 256:
            raise ValueError("Unexpected student input size")
        models[r["run"]] = model.eval().requires_grad_(False)
    features = {name:{arm:torch.empty(len(rows), p["cache_feature_dimension"])
                     for arm in p["feature_heads"]} for name in models}
    started = time.perf_counter()
    batch_size = p["sam_batch_size"]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start+batch_size]
        prepared, sizes = [], []
        for r in chunk:
            with Image.open(r["image"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            resized = transform.apply_image(rgb)
            sizes.append(resized.shape[:2])
            tensor = torch.as_tensor(resized, device=device).permute(2, 0, 1).float()[None]
            prepared.append(sam.preprocess(tensor))
        images = torch.stack([load_image(r["image"], 256, 3) for r in chunk]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            embedding = sam.image_encoder(torch.cat(prepared))
        valid = torch.zeros((len(chunk), 1, 1024, 1024), device=device)
        for j, (h, w) in enumerate(sizes):
            valid[j, :, :h, :w] = 1
        for name, model in models.items():
            logits, bottleneck = unet_features(model, images)
            foreground = (logits > 0).float()
            features[name]["unet"][start:start+len(chunk)] = pool_features(bottleneck, foreground).cpu()
            native_roi = torch.zeros_like(valid)
            for j, (h, w) in enumerate(sizes):
                native_roi[j:j+1, :, :h, :w] = F.interpolate(foreground[j:j+1], size=(h, w),
                                                           mode="bilinear", align_corners=False)
            features[name]["sam"][start:start+len(chunk)] = pool_features(embedding, native_roi, valid).cpu()
        if (start+len(chunk)) % 64 == 0 or start+len(chunk) == len(rows):
            print(which, start+len(chunk), "/", len(rows), "seconds", round(time.perf_counter()-started), flush=True)
    atomic_torch_save({"frozen_sha256":sha256(ROOT/"frozen.json"), "rows":rows,
                       "features":features}, destination)
    write_json(ROOT/f"{which}_cache.json", {"sha256":sha256(destination), "n_images":len(rows),
        "seconds":time.perf_counter()-started, "sam_encoder_passes":len(rows),
        "student_forwards_per_image":len(models), "dtype":"float32_pooled_features"})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze","cache-source","cache-evaluation"])
    parser.add_argument("--output-root", default=str(ROOT))
    args = parser.parse_args()
    ROOT = Path(args.output_root)
    if args.stage == "freeze": freeze()
    else: cache(args.stage.removeprefix("cache-"))
