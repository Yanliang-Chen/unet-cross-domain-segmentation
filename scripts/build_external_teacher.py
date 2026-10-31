"""Source-prompted frozen SAM cache; never reads target annotation paths."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.nn import functional as F
import yaml
from segment_anything import SamPredictor, sam_model_registry

from datasets.files import index_images
from datasets.segmentation import load_image
from scripts.check_protocol import sha256
from scripts.checkpoints import checkpoint_model
from utils import atomic_torch_save, choose_device


def points(probability, image):
    foreground = probability > .5
    labels, n = ndimage.label(foreground, structure=np.ones((3,3)))
    if not n:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    yy, xx = np.where(labels == sizes.argmax())
    center = [float(xx.mean()), float(yy.mean())]
    pixels = image.transpose(1,2,0)
    prototype = pixels[foreground].mean(0)
    deviation = ((pixels-prototype)**2).sum(-1)
    deviation[~foreground] = -1
    y, x = np.unravel_index(deviation.argmax(), deviation.shape)
    return np.asarray([center, [x,y]], dtype=np.float32)


@torch.inference_mode()
def main():
    p = yaml.safe_load(Path("external_teacher_protocol.yaml").read_text(encoding="utf-8"))
    if sha256(p["source_selection"]) != p["source_selection_sha256"]:
        raise ValueError("Source selection changed")
    checkpoint = Path(p["sam_checkpoint"])
    if checkpoint.stat().st_size != p["sam_checkpoint_bytes"]:
        raise ValueError("SAM download incomplete")
    output = Path("runs/real/external_teacher/targets")
    if output.exists():
        raise FileExistsError(output)
    images = index_images(p["target_images"])
    if len(images) != p["target_count"]:
        raise ValueError("Target image set changed")
    selected = json.loads(Path(p["source_selection"]).read_text(encoding="utf-8"))["runs"]
    expected = {f"{a}_seed{s}" for a in p["augmentations"] for s in p["seeds"]}
    if {r["run"] for r in selected} != expected or len(selected) != len(expected):
        raise ValueError("Incomplete source models")
    device = choose_device("cuda")
    torch.set_num_threads(4)
    sam = sam_model_registry[p["sam_model"]](checkpoint=str(checkpoint)).to(device).eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    predictor = SamPredictor(sam)
    models, runs = {}, {}
    for row in selected:
        s = row["methods"]["source"]
        model, size = checkpoint_model(s["checkpoint"], s["checkpoint_sha256"], row["seed"], row["augmentation"], device)
        if size != 256:
            raise ValueError("Unexpected source resolution")
        models[row["run"]] = model.eval()
        runs[row["run"]] = {"run": row["run"], "seed": row["seed"], "augmentation": row["augmentation"],
                            "source": s, "files": []}
        (output/row["run"]).mkdir(parents=True)
    for index, path in enumerate(images):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        x = load_image(path, 256, 3)[None].to(device)
        probabilities = {name: model(x).sigmoid() for name, model in models.items()}
        prompts = {name: points(prob[0,0].cpu().numpy(), x[0].cpu().numpy())
                   for name, prob in probabilities.items()}
        if any(v is not None for v in prompts.values()):
            predictor.set_image(rgb)
        for name, source_probability in probabilities.items():
            prompt = prompts[name]
            quality, agreement = None, None
            foundation_probability = source_probability
            if prompt is not None:
                native = (torch.as_tensor(prompt, device=device)+.5) * torch.tensor(
                    [rgb.shape[1]/256, rgb.shape[0]/256], device=device)
                coordinates = predictor.transform.apply_coords_torch(native, rgb.shape[:2])[None]
                logits, qualities, _ = predictor.predict_torch(point_coords=coordinates,
                    point_labels=torch.ones((1,2), dtype=torch.int64, device=device),
                    multimask_output=True, return_logits=True)
                candidate_probabilities = F.interpolate(logits.sigmoid(), size=(256,256),
                                                        mode="bilinear", align_corners=False)
                candidates = candidate_probabilities > .5
                reference = source_probability > .5
                intersection = (candidates & reference).sum((2,3)).float()
                dice = (2*intersection+1e-6)/(candidates.sum((2,3))+reference.sum((2,3))+1e-6)
                best = max(range(3), key=lambda k: (float(dice[0,k]), float(qualities[0,k]), -k))
                foundation_probability = candidate_probabilities[:, best:best+1]
                quality, agreement = float(qualities[0,best]), float(dice[0,best])
            destination = output/name/f"{path.stem}.pt"
            atomic_torch_save({"source_probability": source_probability[0].cpu().half(),
                "foundation_probability": foundation_probability[0].cpu().half(),
                "valid": prompt is not None}, destination)
            runs[name]["files"].append({"image": str(path.resolve()), "image_sha256": sha256(path),
                "cache": str(destination.resolve()), "cache_sha256": sha256(destination),
                "valid": prompt is not None, "agreement": agreement, "sam_quality": quality})
        if (index+1) % 20 == 0 or index+1 == len(images):
            print("Encoded target images", index+1, "/", len(images), flush=True)
    result = {"protocol_sha256": sha256("external_teacher_protocol.yaml"),
              "generation_script_sha256": sha256(__file__), "sam_checkpoint_sha256": sha256(checkpoint),
              "created_utc": datetime.now(timezone.utc).isoformat(), "runs": list(runs.values())}
    (output/"manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for name, run in runs.items():
        print(name, "valid_prompts", sum(r["valid"] for r in run["files"]), flush=True)


if __name__ == "__main__":
    main()
