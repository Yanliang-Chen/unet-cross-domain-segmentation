# Cross-Domain Biomedical Image Segmentation with U-Net

A cross-platform PyTorch research starter for **Windows + NVIDIA (CUDA)** and **Apple Silicon MacBook (MPS)**. The current implementation is a **binary segmentation U-Net baseline**, not yet the full domain-generalization or label-free model-selection study.

## Project goals
- Reproduce a lightweight U-Net baseline with consistent data and evaluation.
- Compare encoder/network variants on a common source dataset.
- Evaluate out-of-domain medical-image segmentation with Dice and IoU.
- Explore label-free model selection without access to target-domain annotations during selection.

## Install on both computers
Use Python 3.12 and create a **separate environment** on each device. Source files, not virtual environments, are synchronized with GitHub.

**Windows PowerShell**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements-common.txt
```
Install a compatible PyTorch CUDA wheel from https://pytorch.org/get-started/locally/ .

**Apple Silicon Terminal**
```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements-common.txt
./.venv/bin/python -m pip install torch
```
Run `scripts/check_device.py` and confirm CUDA/MPS availability.

## First end-to-end smoke test
From the project root, using the appropriate `.venv` Python:

```bash
python scripts/make_demo_data.py
python train.py --epochs 1
python evaluate.py --checkpoint runs/unet_baseline/latest.pt
```

The generated data consist of **synthetic circles, not medical images**. Any metrics on them are installation checks only.

To resume training with unchanged model and data:
```bash
python train.py --resume runs/unet_baseline/latest.pt --epochs 20
```
Here `--epochs 20` is the **final epoch count**, not 20 more epochs.

## Dataset layout
```text
data/your_dataset/
  train/images/ ...  train/masks/ ...
  val/images/   ...  val/masks/   ...
  test/images/  ...  test/masks/  ...
```
Input files are 8-bit images with matching image/mask stems; masks are binary. For real medical research, split at the **patient level**, prevent leakage, and adapt preprocessing to the modality (e.g. DICOM/NIfTI/16-bit data require dedicated loading).

## Sync strategy
- **GitHub:** code, `config.yaml`, small documentation and experiment definitions.
- **Local disks:** editable project checkout and datasets.
- **Approved cloud storage:** only completed, immutable model checkpoints and appropriately permitted datasets. Don't run training inside an auto-synced folder.
- **Never commit:** raw medical images, patient data, `.venv`, `runs/`, `*.pt`, or secrets.

The training entrypoint selects CUDA, MPS or CPU automatically. CUDA/MPS differences mean bitwise identical training is not guaranteed.

See `README_CN.md` for a Chinese quick-start guide.
