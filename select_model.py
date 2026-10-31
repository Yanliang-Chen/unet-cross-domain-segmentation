"""Select the source-validation-best checkpoint without target labels."""
import argparse
import csv
from pathlib import Path
from scripts.artifacts import write_json
from scripts.check_protocol import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", default="runs/source_selection.json")
    args = parser.parse_args()
    if len(args.runs) != 1:
        parser.error("Select one seed/augmentation run at a time")
    root = Path(args.runs[0])
    with (root/"metrics.csv").open(newline="", encoding="utf-8") as stream:
        candidates = list(csv.DictReader(stream))
    if not candidates:
        raise ValueError("No completed source-validation epochs")
    best = min(candidates, key=lambda row:(-float(row["source_val_dice"]),int(row["epoch"])))
    checkpoint = root/"checkpoints"/f"epoch_{int(best['epoch']):03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    write_json(output, {"method":"source_validation_Dice", "methods":{"source":str(checkpoint.resolve())},
        "selected_checkpoint":str(checkpoint.resolve()), "checkpoint_sha256":sha256(checkpoint),
        "epoch":int(best["epoch"]), "source_val_dice":float(best["source_val_dice"]),
        "target_labels_used":False})
    print("Selected", checkpoint, "source Dice", best["source_val_dice"])


if __name__ == "__main__":
    main()
