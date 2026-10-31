"""Evaluate the matched student arms and a frozen-SAM inference reference."""
import csv
import json
import statistics
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

from datasets.segmentation import load_image, SegmentationDataset
from scripts.check_protocol import sha256
from scripts.checkpoints import checkpoint_model
from datasets.polypgen import center_pairs
from utils import choose_device, per_image_metrics


class Frames(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.rows[index]["image"]
        with Image.open(path) as im:
            width, height = im.size
        return load_image(path,256,3), index, height, width


@torch.inference_mode()
def main():
    root = Path("runs/real/external_teacher")
    if (root/"evaluation.json").exists() or (root/"per_image.csv").exists():
        raise FileExistsError("Evaluation already exists")
    p = yaml.safe_load(Path("external_teacher_protocol.yaml").read_text(encoding="utf-8"))
    evaluation_protocol = Path("external_teacher_evaluation_protocol.yaml")
    evaluation = yaml.safe_load(evaluation_protocol.read_text(encoding="utf-8"))
    manifest = json.loads((root/"adaptation.json").read_text(encoding="utf-8"))
    reference = json.loads((root/"reference/manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol_sha256"] != sha256("external_teacher_protocol.yaml"):
        raise ValueError("Training protocol changed")
    if reference["adaptation_sha256"] != sha256(root/"adaptation.json"):
        raise ValueError("Reference not generated after this adaptation")
    if manifest["teacher_manifest_sha256"] != sha256(root/"targets/manifest.json"):
        raise ValueError("Training teacher changed")
    runs = {(r["run"],r["arm"]): r for r in manifest["runs"]}
    expected = {f"{a}_seed{s}" for a in p["augmentations"] for s in p["seeds"]}
    if len(runs) != 12 or set(runs) != {(r,a) for r in expected for a in p["arms"]}:
        raise ValueError("Incomplete matched arms")
    control_path = Path("runs/real/polypgen_source_control/adaptation.json")
    controls = {r["run"]:r for r in json.loads(control_path.read_text(encoding="utf-8"))["runs"]}
    references = {r["run"]:{f["sample_id"]:f for f in r["files"]} for r in reference["runs"]}
    with Path("data/polyp_png/split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        groups = {r["sample_id"]:r["group_id"] for r in csv.DictReader(stream) if r["split"] == "target_test"}
    clinic = SegmentationDataset("data/polyp_png", "target_test", 256, 3).pairs
    if len(clinic) != 427 or len(set(groups.values())) != 20:
        raise ValueError("Clinic split changed")
    domains = {"CVC-ClinicDB":clinic, **{f"PolypGen-{k}":v for k,v in
        center_pairs().items()}}
    if set(domains) != set(evaluation["domains"]):
        raise ValueError("Unexpected evaluation domain")
    rows = [{"domain":domain,"image":str(image),"mask":str(mask),"sample_id":image.stem,
             "group":groups[image.stem] if domain == "CVC-ClinicDB" else ""}
            for domain,pairs in domains.items() for image,mask in pairs]
    loader = DataLoader(Frames(rows), batch_size=16, shuffle=False)
    device = choose_device("cuda")
    torch.set_num_threads(4)
    results = []
    with (root/"per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run","augmentation","seed","domain","sample_id",
            "group","method","gt_fraction","dice","iou","precision","recall","foreground_fraction"])
        writer.writeheader()
        for name in sorted(expected):
            external, self_run = runs[(name,"external")],runs[(name,"self")]
            source = external["source"]
            if any(r["source"]["checkpoint_sha256"] != source["checkpoint_sha256"]
                   for r in (self_run,controls[name])):
                raise ValueError("Mismatched source parents")
            choices = {"source":(source["checkpoint"],source["checkpoint_sha256"])}
            for method,r in (("external",external),("self",self_run),("control",controls[name])):
                choices[method] = (r["best_checkpoint"],r["best_checkpoint_sha256"])
            models = {}
            for method,(path,digest) in choices.items():
                model,size = checkpoint_model(path,digest,external["seed"],external["augmentation"],device)
                if size != 256:
                    raise ValueError("Student size changed")
                models[method] = model.eval()
            values = {}
            for batch,indices,heights,widths in loader:
                batch = batch.to(device)
                predictions = {m:model(batch).sigmoid() for m,model in models.items()}
                for j,index in enumerate(indices.tolist()):
                    row = rows[index]
                    size = (int(heights[j]),int(widths[j]))
                    native = {m:F.interpolate(v[j:j+1],size=size,mode="bilinear",align_corners=False)
                              for m,v in predictions.items()}
                    for method in ("external","self","control"):
                        native[method+"_ensemble"] = (native["source"]+native[method])/2
                    if row["domain"] == "CVC-ClinicDB":
                        ref = references[name][row["sample_id"]]
                        if sha256(ref["cache"]) != ref["cache_sha256"] or sha256(row["image"]) != ref["image_sha256"]:
                            raise ValueError("Frozen reference changed")
                        probability = torch.load(ref["cache"],map_location="cpu",weights_only=True)["probability"]
                        native["sam"] = F.interpolate(probability.float()[None].to(device),size=size,
                                                       mode="bilinear",align_corners=False)
                        native["source_sam_ensemble"] = (native["source"]+native["sam"])/2
                    with Image.open(row["mask"]) as im:
                        target = torch.from_numpy((np.asarray(im.convert("L"))>127).copy())
                    if tuple(target.shape) != size:
                        raise ValueError("Native mask differs")
                    target = target[None,None].to(device=device,dtype=torch.float32)
                    fraction = float(target.mean())
                    for method,probability in native.items():
                        metric = per_image_metrics(torch.logit(probability.clamp(1e-6,1-1e-6)),target)[0]
                        item = {"run":name,"augmentation":external["augmentation"],"seed":external["seed"],
                            "domain":row["domain"],"sample_id":row["sample_id"],"group":row["group"],
                            "method":method,"gt_fraction":fraction,**metric}
                        writer.writerow(item)
                        values.setdefault((row["domain"],method),[]).append(item)
                if int(indices[-1]) % 512 < 16:
                    print(name,"images",int(indices[-1])+1,"/",len(rows),flush=True)
            for (domain,method),items in values.items():
                small = [r["dice"] for r in items if 0<r["gt_fraction"]<.02]
                empty = [r["foreground_fraction"]>0 for r in items if r["gt_fraction"] == 0]
                grouped = {}
                for r in items:
                    if r["group"]:
                        grouped.setdefault(r["group"],[]).append(r["dice"])
                results.append({"run":name,"augmentation":external["augmentation"],"seed":external["seed"],
                    "domain":domain,"method":method,"n":len(items),
                    "dice":statistics.mean(r["dice"] for r in items),
                    "precision":statistics.mean(r["precision"] for r in items),
                    "recall":statistics.mean(r["recall"] for r in items),
                    "small_positive_dice":statistics.mean(small) if small else None,
                    "empty_fpr":statistics.mean(empty) if empty else None,
                    "group_mean_dice":statistics.mean(statistics.mean(v) for v in grouped.values()) if grouped else None})
            stream.flush()
            print("Completed",name,flush=True)
            del models
    (root/"evaluation.json").write_text(json.dumps({"status":p["status"],
        "training_protocol_sha256":sha256("external_teacher_protocol.yaml"),
        "evaluation_protocol_sha256":sha256(evaluation_protocol),
        "adaptation_sha256":sha256(root/"adaptation.json"),"control_manifest_sha256":sha256(control_path),
        "reference_manifest_sha256":sha256(root/"reference/manifest.json"),"results":results},indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
