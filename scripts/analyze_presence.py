"""Joint false-positive reduction and positive-Dice preservation assessment."""
import csv
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
from scripts.check_protocol import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default="runs/real/presence_clean")
    parser.add_argument("--protocol",default="presence_evaluation_protocol.yaml")
    args = parser.parse_args()
    root = Path(args.root)
    path = Path(args.protocol)
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    complete = json.loads((root/"evaluation_complete.json").read_text(encoding="utf-8"))
    if complete["evaluation_protocol_sha256"] != sha256(path) or complete["per_image_sha256"] != sha256(root/"per_image.csv"):
        raise ValueError("Frozen evaluation changed")
    scores = defaultdict(list)
    frame_counts = defaultdict(int)
    image_scores = defaultdict(list)
    subgroups = defaultdict(list)
    with (root/"per_image.csv").open(newline="",encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            domain = r["domain"]
            value = float(r["dice"]) if not domain.startswith("normal_") else float(r["foreground_fraction"])>0
            key = (r["augmentation"],r["seed"],domain,r["group"],r["method"])
            scores[key].append(value)
            image_scores[(r["augmentation"],domain,r["method"])].append(value)
            area = float(r["gt_fraction"])
            if not domain.startswith("normal_") and area > 0:
                subgroups[(r["augmentation"],domain,r["method"],"positive_only_dice")].append(float(r["dice"]))
                if area < .02:
                    subgroups[(r["augmentation"],domain,r["method"],"small_positive_dice")].append(float(r["dice"]))
            if domain.startswith("normal_"):
                subgroups[(r["augmentation"],domain,r["method"],"foreground_fraction")].append(float(r["foreground_fraction"]))
            if r["augmentation"] == "a0" and r["seed"] == "42" and r["method"] == "external_previous":
                frame_counts[(domain,r["group"])] += 1
    means = {k:float(np.mean(v)) for k,v in scores.items()}
    out = {"status":p.get("scope","video_group_holdout_not_patient_level_confirmation"),"primary":{},
           "comparisons":{},"group_means":{},"frame_equal_means":{
               "/".join(k):float(np.mean(v)) for k,v in image_scores.items()},"groups":{
               "/".join(k):v for k,v in frame_counts.items()},"subgroups":{
               "/".join(k):float(np.mean(v)) for k,v in subgroups.items()}}
    pairs = [(p["primary_method"],p["primary_comparator"]),
             *(tuple(pair) for pair in p["secondary_comparisons"])]
    for aug in p["augmentations"]:
        for domain in ("CVC-ClinicDB","normal_holdout","normal_diagnostic","PolypGen-C6"):
            groups = sorted({k[3] for k in means if k[0] == aug and k[2] == domain})
            for method in sorted({k[4] for k in means if k[0] == aug and k[2] == domain}):
                out["group_means"][f"{aug}/{domain}/{method}"] = float(np.mean([
                    means[(aug,str(s),domain,g,method)] for s in p["seeds"] for g in groups]))
            for method,baseline in pairs:
                differences = np.array([np.mean([
                    means[(aug,str(s),domain,g,method)]-means[(aug,str(s),domain,g,baseline)]
                    for s in p["seeds"]]) for g in groups])
                rng = np.random.default_rng(p["bootstrap_seed"])
                bootstrap = differences[rng.integers(0,len(differences),(p["bootstrap_replicates"],len(differences)))].mean(1)
                out["comparisons"][f"{aug}/{domain}/{method}_minus_{baseline}"] = {
                    "n_groups":len(groups),"difference":float(differences.mean()),
                    "ci95":np.quantile(bootstrap,[.025,.975]).tolist() if groups != [""] else None,
                    "per_group_difference":dict(zip(groups,differences.tolist())),
                    "per_seed_difference":{str(s):float(np.mean([
                        means[(aug,str(s),domain,g,method)]-means[(aug,str(s),domain,g,baseline)] for g in groups]))
                        for s in p["seeds"]}}
        key = f"{p['primary_method']}_minus_{p['primary_comparator']}"
        positive = out["comparisons"][f"{aug}/CVC-ClinicDB/{key}"]
        negative = out["comparisons"][f"{aug}/normal_holdout/{key}"]
        rules = p["success_requires_both_augmentations"]
        reduction = -negative["difference"]
        reduction_ci = [-negative["ci95"][1],-negative["ci95"][0]]
        passed = (reduction >= rules["negative_FPR_reduction_at_least"]
                  and reduction_ci[0] > rules["negative_FPR_reduction_95ci_lower_above"]
                  and positive["ci95"][0] >= rules["positive_Dice_difference_95ci_lower_at_least"])
        out["primary"][aug] = {"positive_difference":positive["difference"],"positive_ci95":positive["ci95"],
            "negative_FPR_reduction":reduction,"reduction_ci95":reduction_ci,"passed":passed}
    out["primary_joint_passed"] = all(r["passed"] for r in out["primary"].values())
    (root/"analysis.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"primary":out["primary"],"joint_passed":out["primary_joint_passed"]},indent=2))
    for aug in p["augmentations"]:
        for domain in ("CVC-ClinicDB","normal_holdout","PolypGen-C6"):
            print(aug,domain,{k.split("/")[-1]:round(v,4) for k,v in out["group_means"].items()
                             if k.startswith(f"{aug}/{domain}/")})


if __name__ == "__main__":
    main()
