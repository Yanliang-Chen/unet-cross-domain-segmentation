"""Run the frozen A0/A1 by three-seed matrix without touching target_test."""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.check_protocol import check_protocol


def commands(protocol_path, config_path, data_dir, run_root, device):
    protocol = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
    for augmentation in protocol["augmentations"]:
        for seed in protocol["seeds"]:
            name = f"{augmentation}_seed{seed}"
            run = Path(run_root) / name
            selection = Path(run_root) / "selections" / f"{name}.json"
            train = [sys.executable, "train.py", "--config", str(config_path),
                     "--protocol", str(protocol_path), "--data-dir", str(data_dir),
                     "--run-dir", str(run), "--seed", str(seed),
                     "--augmentation", augmentation, "--device", device]
            select = [sys.executable, "select_model.py", "--runs", str(run),
                      "--output", str(selection)]
            yield name, run, selection, train, select


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="protocol.yaml")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="data/polyp_png")
    parser.add_argument("--run-root", default="runs/real")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", help="One cell, e.g. a0_seed42")
    args = parser.parse_args()
    matrix = list(commands(args.protocol, args.config, args.data_dir, args.run_root, args.device))
    if args.only:
        matrix = [cell for cell in matrix if cell[0] == args.only]
        if not matrix:
            parser.error(f"Unknown matrix cell: {args.only}")
    if not args.dry_run:
        check_protocol(args.protocol, args.data_dir)
    for name, run, selection, train, select in matrix:
        print(f"[{name}]", flush=True)
        if not args.dry_run and selection.exists():
            print(f"Selection already exists; skipping completed cell: {selection}", flush=True)
            continue
        for command in (train, select):
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            if command is train and run.exists() and any(run.iterdir()):
                if not (run / "latest.pt").exists():
                    raise FileExistsError(f"{run} is nonempty without latest.pt; inspect it manually")
                command = command + ["--resume", str(run / "latest.pt")]
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
