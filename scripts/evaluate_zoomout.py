"""Exploratory native-mask comparison of zoom-out augmentation and frozen A1."""
from scripts.evaluate_tversky import compare


if __name__ == "__main__":
    compare("zoomout_protocol.yaml", "runs/real/zoomout",
            "runs/real/zoomout/evaluation.json", "a3", "bce_dice", "zoomout")
