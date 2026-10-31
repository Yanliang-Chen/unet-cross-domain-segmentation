"""Paired video-group analysis of the predeclared development comparisons."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from scripts.check_protocol import sha256


def main():
    root = Path("runs/real/external_teacher")
    p = yaml.safe_load(Path("external_teacher_evaluation_protocol.yaml").read_text(encoding="utf-8"))
    evaluation = json.loads((root/"evaluation.json").read_text(encoding="utf-8"))
    if evaluation["evaluation_protocol_sha256"] != sha256("external_teacher_evaluation_protocol.yaml"):
        raise ValueError("Evaluation protocol changed")
    scores = defaultdict(list)
    with (root/"per_image.csv").open(newline="",encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if r["domain"] == "CVC-ClinicDB":
                scores[(r["augmentation"],r["seed"],r["group"],r["method"])].append(float(r["dice"]))
    means = {k:float(np.mean(v)) for k,v in scores.items()}
    result = {"status":"controlled_development_not_independent_confirmation",
              "evaluation_sha256":sha256(root/"evaluation.json"),"comparisons":{},"aggregate":{}}
    pairs = [("external_ensemble","control_ensemble"),("external_ensemble","self_ensemble"),
             ("external","control"),("sam","control_ensemble"),("source_sam_ensemble","control_ensemble")]
    for a in ("a0","a1"):
        groups = sorted({k[2] for k in means if k[0] == a})
        if len(groups) != 20:
            raise ValueError("Video groups changed")
        for method,baseline in pairs:
            diff = np.array([np.mean([means[(a,str(seed),g,method)]-means[(a,str(seed),g,baseline)]
                                     for seed in (42,43,44)]) for g in groups])
            rng = np.random.default_rng(p["bootstrap_seed"])
            boots = diff[rng.integers(0,len(diff),(p["bootstrap_replicates"],len(diff)))].mean(1)
            result["comparisons"][f"{a}/{method}_minus_{baseline}"] = {
                "video_equal_delta":float(diff.mean()),"ci95":np.quantile(boots,[.025,.975]).tolist(),
                "per_seed_video_equal_delta":{str(s):float(np.mean([
                    means[(a,str(s),g,method)]-means[(a,str(s),g,baseline)] for g in groups])) for s in (42,43,44)}}
        rows = [r for r in evaluation["results"] if r["augmentation"] == a]
        for domain in sorted({r["domain"] for r in rows}):
            result["aggregate"][f"{a}/{domain}"] = {}
            subset = [r for r in rows if r["domain"] == domain]
            for method in sorted({r["method"] for r in subset}):
                found = [r for r in subset if r["method"] == method]
                result["aggregate"][f"{a}/{domain}"][method] = {
                    k:statistics.mean(r[k] for r in found) if all(r[k] is not None for r in found) else None
                    for k in ("dice","precision","recall","small_positive_dice","empty_fpr","group_mean_dice")}
    (root/"analysis.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for a in ("a0","a1"):
        print(a,"CVC", {m:round(r["dice"],4) for m,r in result["aggregate"][f"{a}/CVC-ClinicDB"].items()})
        for method,baseline in pairs:
            r = result["comparisons"][f"{a}/{method}_minus_{baseline}"]
            print(method,"minus",baseline,round(r["video_equal_delta"],4),"CI",[round(x,4) for x in r["ci95"]])
    print("Saved",root/"analysis.json")


if __name__ == "__main__":
    main()
