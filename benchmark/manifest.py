"""Reproducibility records kept in git (amendment v2, phase 4).

Usage:
  python -m benchmark.manifest data        # results/manifests/<scenario>.csv, archives.csv, environment.txt
  python -m benchmark.manifest deployed    # results/manifests/deployed_models.csv, results/per_image/<target>.csv.gz
  python -m benchmark.manifest verify      # recompute a paired gain from the per-image files alone
No images, masks or weights are written; only paths relative to data/, groups and SHA-256 hashes.
"""
import argparse
import csv
import gzip
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

from benchmark.analyze import load_pools
from benchmark.data import ROOT, SCENARIOS, scenario_rows
from benchmark.report import POOLS, RESULTS, base_run, trajectories
from benchmark.score import SCORES

OUT = RESULTS.parent/"manifests"
PER_IMAGE = RESULTS.parent/"per_image"
TARGETS = ("clinicdb", "etis", "colondb", "polypgen", "far", "polypdb")
ARCHIVES = ("kvasir-seg.zip", "CVC/CVC-ClinicDB_datasets.zip", "TestDataset.zip",
            "PolypGen2021_MultiCenterData_v3.zip", "FAR-POLYP-SEG/Dataset.zip", "PolypDB/PolypDB.zip")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix() if path else ""


def data_manifests():
    OUT.mkdir(parents=True, exist_ok=True)
    source_train = [{"image": str(p), "mask": str(ROOT/"polyp_png/source_train/masks"/p.name), "kind": "polyp",
                     "group": "", "role": "train"}
                    for p in sorted((ROOT/"polyp_png/source_train/images").glob("*"))]
    for scenario, rows in [("source_train", source_train), *((s, scenario_rows(s)) for s in SCENARIOS)]:
        with (OUT/f"{scenario}.csv").open("w", newline="", encoding="utf-8") as stream:
            w = csv.writer(stream)
            w.writerow(["index", "image", "mask", "kind", "group", "role", "image_sha256"])
            for i, r in enumerate(rows):
                w.writerow([i, relative(r["image"]), relative(r["mask"]), r["kind"], r["group"], r["role"], sha256(r["image"])])
        print(scenario, len(rows), flush=True)
    with (OUT/"archives.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["archive", "bytes", "sha256"])
        for name in ARCHIVES:
            path = ROOT/name
            w.writerow([name, path.stat().st_size, sha256(path)] if path.exists() else [name, "", "missing"])
    import torch
    lines = [f"python {sys.version.split()[0]}", f"platform {platform.platform()}", f"torch {torch.__version__}",
             f"cuda {torch.version.cuda}", f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
             "", "pip freeze:", subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout]
    (OUT/"environment.txt").write_text("\n".join(lines), encoding="utf-8")


def deployed():
    table, _ = load_pools(POOLS, features=False)
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    hashes, rows = {}, []
    for target in TARGETS:
        picks = {k: max(v, key=lambda r: r["validators"]["source_risk"]) for k, v in trajectories(table[target]).items()}
        with gzip.open(PER_IMAGE/f"{target}.csv.gz", "wt", newline="", encoding="utf-8") as stream:
            w = csv.writer(stream)
            w.writerow(["id", "index", "dice", "fg_native"])
            for key, r in sorted(picks.items()):
                scores = np.load(SCORES/r["pool"]/target/f"{r['id']}.npz")
                for i, (d, f) in enumerate(zip(scores["dice"], scores["fg_native"])):
                    w.writerow([r["id"], i, "" if np.isnan(d) else f"{d:.5f}", f"{f:.6f}"])
                checkpoint = r["checkpoints"][0]
                hashes.setdefault(checkpoint, sha256(checkpoint))
                rows.append([target, r["algorithm"], r["lam"], base_run(r), r["step"], r["id"],
                             Path(checkpoint).as_posix(), hashes[checkpoint],
                             f"{r['validators']['source_risk']:.5f}", f"{r['dice']:.5f}"])
        print(target, len(picks), flush=True)
    with (OUT/"deployed_models.csv").open("w", newline="", encoding="utf-8") as stream:
        w = csv.writer(stream)
        w.writerow(["target", "algorithm", "lam", "base_run", "step", "id", "checkpoint", "checkpoint_sha256",
                    "source_risk", "dice"])
        w.writerows(rows)


def verify(target="clinicdb", algorithm="dacs", reference="source_only"):
    """Point estimate of the paired Dice gain from git-tracked files only (no runs/, no GPU)."""
    meta = {}
    with (OUT/f"{target}.csv").open(encoding="utf-8") as stream:
        meta = {int(r["index"]): r for r in csv.DictReader(stream)}
    models = {}
    with (OUT/"deployed_models.csv").open(encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if r["target"] == target:
                models[r["id"]] = r
    dice = {}
    with gzip.open(PER_IMAGE/f"{target}.csv.gz", "rt", encoding="utf-8") as stream:
        for r in csv.DictReader(stream):
            if r["dice"]:
                dice.setdefault(r["id"], {})[int(r["index"])] = float(r["dice"])
    reference_by_run = {m["base_run"]: m["id"] for m in models.values() if m["algorithm"] == reference}
    index = sorted(dice[next(iter(reference_by_run.values()))])
    groups = np.array([meta[i]["group"] for i in index])
    labels, inverse = np.unique(groups, return_inverse=True) if all(groups) else (np.arange(len(index)), np.arange(len(index)))
    per_run = {}
    for m in models.values():
        if m["algorithm"] == algorithm:
            diff = np.array([dice[m["id"]][i] - dice[reference_by_run[m["base_run"]]][i] for i in index])
            per_run.setdefault(m["base_run"], []).append(np.bincount(inverse, diff, len(labels)) / np.bincount(inverse))
    gain = float(np.mean([np.mean(v, 0) for v in per_run.values()]))
    print(f"{target}: {algorithm} - {reference} paired Dice gain from tracked files = {gain:+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["data", "deployed", "verify"])
    args = parser.parse_args()
    {"data": data_manifests, "deployed": deployed, "verify": verify}[args.command]()


if __name__ == "__main__":
    main()
