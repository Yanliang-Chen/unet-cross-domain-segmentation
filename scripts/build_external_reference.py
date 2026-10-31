"""Frozen SAM inference reference on the CVC evaluation images, without masks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from segment_anything import SamPredictor, sam_model_registry
import yaml

from datasets.files import index_images
from datasets.segmentation import load_image
from scripts.build_external_teacher import points
from scripts.check_protocol import sha256
from scripts.checkpoints import checkpoint_model
from utils import choose_device, atomic_torch_save


@torch.inference_mode()
def main():
    protocol = Path("external_teacher_protocol.yaml")
    p = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    root = Path("runs/real/external_teacher")
    adaptation = json.loads((root/"adaptation.json").read_text(encoding="utf-8"))
    if adaptation["protocol_sha256"] != sha256(protocol):
        raise ValueError("Adaptation must be frozen before reference inference")
    manifest = json.loads((root/"targets/manifest.json").read_text(encoding="utf-8"))
    if sha256(p["sam_checkpoint"]) != manifest["sam_checkpoint_sha256"]:
        raise ValueError("SAM checkpoint changed")
    output = root/"reference"
    if output.exists():
        raise FileExistsError(output)
    images = index_images("data/polyp_png/target_test/images")
    if len(images) != 427:
        raise ValueError("Evaluation images changed")
    device = choose_device("cuda")
    torch.set_num_threads(4)
    sam = sam_model_registry[p["sam_model"]](checkpoint=p["sam_checkpoint"]).to(device).eval()
    for parameter in sam.parameters():
        parameter.requires_grad_(False)
    predictor = SamPredictor(sam)
    models, runs = {}, {}
    for run in manifest["runs"]:
        source = run["source"]
        model, size = checkpoint_model(source["checkpoint"], source["checkpoint_sha256"],
                                       run["seed"], run["augmentation"], device)
        if size != 256:
            raise ValueError("Source size changed")
        models[run["run"]] = model.eval()
        runs[run["run"]] = {"run": run["run"], "source": source, "files": []}
        (output/run["run"]).mkdir(parents=True)
    for i, path in enumerate(images):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        image = load_image(path, 256, 3)[None].to(device)
        source_probabilities = {name: model(image).sigmoid() for name, model in models.items()}
        prompts = {name: points(ps[0,0].cpu().numpy(), image[0].cpu().numpy())
                   for name, ps in source_probabilities.items()}
        if any(prompt is not None for prompt in prompts.values()):
            predictor.set_image(rgb)
        for name, source in source_probabilities.items():
            prompt = prompts[name]
            probability = source
            if prompt is not None:
                coords = (torch.as_tensor(prompt, device=device)+.5)*torch.tensor(
                    [rgb.shape[1]/256, rgb.shape[0]/256], device=device)
                coords = predictor.transform.apply_coords_torch(coords, rgb.shape[:2])[None]
                logits, qualities, _ = predictor.predict_torch(point_coords=coords,
                    point_labels=torch.ones((1,2), dtype=torch.int64, device=device),
                    multimask_output=True, return_logits=True)
                native_probabilities = logits.sigmoid()
                candidates = F.interpolate(native_probabilities, size=(256,256), mode="bilinear", align_corners=False)
                masks, reference = candidates > .5, source > .5
                dice = (2*(masks & reference).sum((2,3)).float()+1e-6)/(
                    masks.sum((2,3))+reference.sum((2,3))+1e-6)
                best = max(range(3), key=lambda k: (float(dice[0,k]), float(qualities[0,k]), -k))
                probability = native_probabilities[:,best:best+1]
            path_out = output/name/f"{path.stem}.pt"
            atomic_torch_save({"probability": probability[0].cpu().half()}, path_out)
            runs[name]["files"].append({"sample_id": path.stem, "cache": str(path_out.resolve()),
                "cache_sha256": sha256(path_out), "image_sha256": sha256(path), "valid_prompt": prompt is not None})
        if (i+1) % 50 == 0 or i+1 == len(images):
            print("Frozen inference reference", i+1, "/", len(images), flush=True)
    (output/"manifest.json").write_text(json.dumps({"role": "inference_reference_never_training_input",
        "adaptation_sha256": sha256(root/"adaptation.json"), "protocol_sha256": sha256(protocol),
        "runs": list(runs.values())}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
