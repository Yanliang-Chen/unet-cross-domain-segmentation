"""Retained relative-presence heads using frozen image-feature caches."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
import yaml
from models.presence_gate import fit_head, presence_logits, positive_threshold
from scripts.artifacts import read_json, write_json
from scripts.check_protocol import sha256
from utils import atomic_torch_save


def relative_features(x):
    global_mean, _, foreground, background = x.chunk(4, dim=1)
    foreground = F.normalize(foreground, dim=1)
    return torch.cat((foreground-F.normalize(background, dim=1),
                      foreground-F.normalize(global_mean, dim=1)), 1)


def load_cache(root, which):
    path = root/f"{which}_features.pt"
    if sha256(path) != read_json(root/f"{which}_cache.json")["sha256"]:
        raise ValueError("Frozen feature cache changed")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["frozen_sha256"] != sha256(root/"frozen.json"):
        raise ValueError("Features and frozen image/model identities differ")
    return data


def fit(p, cache_root, output, protocol):
    if output.exists():
        raise FileExistsError(output)
    frozen = read_json(cache_root/"frozen.json")
    manifest = read_json(p["segmentation_manifest"])
    if sha256(p["segmentation_manifest"]) != p["segmentation_manifest_sha256"]:
        raise ValueError("Frozen external students changed")
    students = {r["run"]:r["best_checkpoint_sha256"] for r in manifest["runs"] if r["arm"]=="external"}
    if {r["run"]:r["best_checkpoint_sha256"] for r in frozen["runs"]} != students:
        raise ValueError("Feature parents do not match the baseline experiment")
    data = load_cache(cache_root,"source")
    rows = data["rows"]
    expected = {f"{a}_seed{s}" for a in p["augmentations"] for s in p["seeds"]}
    if set(data["features"]) != expected:
        raise ValueError("Incomplete seed/augmentation cache")
    if dict(Counter(r["role"] for r in rows)) != p["source_counts"]:
        raise ValueError("Source roles changed")
    idx = [i for i,r in enumerate(rows) if r["role"] in p["source_class_weights"]]
    cal = [i for i,r in enumerate(rows) if r["role"]=="positive_calibration"]
    counts = Counter(rows[i]["role"] for i in idx)
    groups = Counter(rows[i]["group"] for i in idx if rows[i]["role"]=="negative")
    weights = torch.tensor([p["source_class_weights"][rows[i]["role"]]/
        (len(groups)*groups[rows[i]["group"]] if rows[i]["role"]=="negative" else counts[rows[i]["role"]]) for i in idx])
    labels = torch.tensor([rows[i]["label"] for i in idx])
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    records = []
    for name in sorted(expected):
        for arm in p["feature_heads"]:
            x = relative_features(data["features"][name][arm])
            head = fit_head(x[idx],labels,weights,p)
            scores = presence_logits(x[cal],head)
            threshold,k = positive_threshold(scores,p["positive_rejection_budget"])
            head["threshold"] = threshold
            path = output/f"{name}_{arm}_head.pt"
            atomic_torch_save(head,path)
            records.append({"run":name,"arm":arm,"head":str(path.resolve()),"head_sha256":sha256(path),
                            "threshold":threshold,"calibration_k":k,"source_positive_retained":int((scores>=threshold).sum())})
    write_json(output/"gates.json",{"protocol_sha256":sha256(protocol),"source_cache_sha256":sha256(cache_root/"source_features.pt"),
        "feature_code_sha256":sha256(__file__),"model_code_sha256":sha256("models/presence_gate.py"),"runs":records})
    print("Fixed",len(records),"relative-presence heads")


def evaluate(p, cache_root, output, protocol):
    if (output/"per_image.csv").exists():
        raise FileExistsError("Output already evaluated")
    gates = read_json(output/"gates.json")
    if gates["protocol_sha256"]!=sha256(protocol) or gates["source_cache_sha256"]!=sha256(cache_root/"source_features.pt"):
        raise ValueError("Head protocol or training features changed")
    if gates["feature_code_sha256"]!=sha256(__file__) or gates["model_code_sha256"]!=sha256("models/presence_gate.py"):
        raise ValueError("Relative feature implementation changed after fitting")
    source, evaluation = load_cache(cache_root,"source"),load_cache(cache_root,"evaluation")
    val = [i for i,r in enumerate(source["rows"]) if r["role"]=="negative_validation"]
    decisions = {}
    for r in gates["runs"]:
        if sha256(r["head"])!=r["head_sha256"]:
            raise ValueError("Frozen head changed")
        head = torch.load(r["head"],map_location="cpu",weights_only=True)
        for data,indices in ((source,val),(evaluation,list(range(len(evaluation["rows"]))))):
            scores = presence_logits(relative_features(data["features"][r["run"]][r["arm"]][indices]),head)
            for i,score in zip(indices,scores.tolist()):
                row = data["rows"][i]
                decisions[r["run"],r["arm"],row["domain"],row["group"],row["sample_id"]] = score>=head["threshold"]
    baseline = Path(p["baseline_csv"])
    if sha256(baseline)!=p["baseline_csv_sha256"]:
        raise ValueError("Frozen baseline outcomes changed")
    used = set()
    with baseline.open(newline="",encoding="utf-8") as old,(output/"per_image.csv").open("w",newline="",encoding="utf-8") as out:
        reader = csv.DictReader(old)
        writer = csv.DictWriter(out,fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["method"] not in {"source_previous","self_previous","external_previous"}:
                continue
            writer.writerow(row)
            if row["method"]!="external_previous":
                continue
            for arm in p["feature_heads"]:
                key = row["run"],arm,row["domain"],row["group"],row["sample_id"]
                if key in used:
                    raise ValueError("Duplicate image outcome")
                used.add(key)
                result = {**row,"method":arm+"_relative_gate"}
                if not decisions[key]:
                    empty = float(row["gt_fraction"])==0
                    result.update(dice=float(empty),iou=float(empty),precision=float(empty),recall=float(empty),foreground_fraction=0.)
                writer.writerow(result)
    if used!=set(decisions):
        raise ValueError("Incomplete image coverage")
    write_json(output/"evaluation_complete.json",{"evaluation_protocol_sha256":sha256("presence_evaluation_protocol.yaml"),
        "per_image_sha256":sha256(output/"per_image.csv"),"gates_sha256":sha256(output/"gates.json")})
    print("Evaluated all",len(used),"relative-head image outcomes")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage",choices=["fit","evaluate"])
    parser.add_argument("--cache-root",default="runs/real/presence_gate")
    parser.add_argument("--output-root",default="runs/real/presence_clean")
    parser.add_argument("--protocol",default="presence_protocol.yaml")
    args = parser.parse_args()
    p = yaml.safe_load(Path(args.protocol).read_text(encoding="utf-8"))
    {"fit":fit,"evaluate":evaluate}[args.stage](p,Path(args.cache_root),Path(args.output_root),args.protocol)
