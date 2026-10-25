# Robust Cross-Domain Biomedical Image Segmentation with U-Net and Label-Free Model Selection

A small, reproducible research starting point for **binary polyp segmentation**. Train a standard U-Net on labeled Kvasir-SEG images, use **unlabeled CVC-ClinicDB images** to choose one candidate checkpoint, and only then evaluate on held-out CVC-ClinicDB masks. A shallower U-Net is available as an optional architecture comparison.

The proposed full experiment, including how the segmentation task relates to Xiong et al.'s classification datasets, is in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). It distinguishes planned analyses from features already implemented.

## Research question

Does selecting a U-Net checkpoint from unlabeled target images improve held-out target Dice compared with choosing the checkpoint with the best **source validation Dice**? Here the unsupervised score measures prediction stability under horizontal flip and mild brightness change. Source validation Dice restricts the candidate pool to within 0.03 of the best source candidate. This is a deliberately simple baseline, **not** an implementation of Xiong et al.'s two-level ensemble/reference method and **not** a validated estimator of segmentation quality. Stable empty predictions remain a risk. Run the source-only baseline and inspect the predictions before claiming an improvement.

Xiong et al., *Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging* (2026), use ADNI/AIBL MRI and chest X-ray **classification** datasets. They identify segmentation as future work. Those datasets do not directly supply the pixel masks this project needs. This repository borrows the label-free selection **problem framing** and studies it on paired segmentation data instead.

## Data and protocol

- **Labeled source:** [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/), 1,000 endoscopy image/mask pairs; research/education use and original paper citation required.
- **Unlabeled selection / held-out target:** [CVC-ClinicDB](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/), 612 frames and corresponding polyp masks. Its frames come from multiple videos: use video/sequence IDs for splitting when possible.
- Source `source_train`: model fitting; source `source_val`: source baseline and guardrail.
- Target `target_select/images`: checkpoint scoring only; no masks staged here.
- Target `target_test/images+masks`: final evaluation only. **Never use test Dice to revise the selection rule, thresholds or architecture** for the same report.

Both source and target images are resized to 256 x 256 for this baseline. Masks use nearest-neighbor resize and threshold >127 (especially relevant for JPEG masks). This square resize distorts aspect ratio; use a single consistent preprocessing policy for comparison and report this limitation. Metrics are per-image mean Dice and IoU, thresholding predictions at 0.5. Dataset access terms must be followed; images, masks, and checkpoints are ignored by Git.

### Expected raw layout

Download/extract the datasets yourself into `data/raw/`:

```text
data/raw/kvasir-seg/images/<matching-stem>.jpg
data/raw/kvasir-seg/masks/<matching-stem>.jpg
data/raw/cvc-clinicdb/Original/<matching-stem>.tif
data/raw/cvc-clinicdb/Ground Truth/<matching-stem>.tif
data/raw/cvc-clinicdb/metadata.csv  # optional: frame_id,sequence_id
```

CVC-ClinicDB can also use `images/` and `masks/` folder names. Its metadata CSV may instead have `sample_id,group_id`. The script does **not** download or redistribute medical images.

### Environment (Windows NVIDIA or Apple Silicon Mac)

Use Python 3.10-3.12 and create a separate local environment. On Windows PowerShell:

```powershell
./scripts/setup_windows.ps1
# If PowerShell script execution is disabled, use the commands in the script manually.
.\.venv\Scripts\python.exe -m scripts.check_device
```

On macOS:

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
python -m scripts.check_device
```

The default `--device auto` uses CUDA, then MPS, then CPU. `amp_cuda` only activates mixed precision on CUDA. Keep datasets in `data/` on each machine; sync **code and small run summaries** via Git. Copy a checkpoint with a file transfer tool if you want to resume it on the other platform; never Git commit patient images or model weights. Floating point results can differ across devices.

### Prepare real datasets

For video-aware target splitting, supply a CSV mapping every target filename stem to a video/sequence ID:

```csv
sample_id,group_id
1,video_01
2,video_01
26,video_02
```

Or use `frame_id,sequence_id` columns from a dataset distribution, after checking that IDs match image filenames. Then run:

```bash
python -m scripts.prepare_polyp_data --source data/raw/kvasir-seg --target data/raw/cvc-clinicdb --target-groups-csv data/raw/cvc-clinicdb/metadata.csv
```

If you lack sequence identifiers, omit `--target-groups-csv` for an **exploratory** image-level split only; adjacent video frames may appear in both target partitions and overstate generalization. The script refuses to overwrite a populated output folder. By default: 80/20 source train/validation, 30/70 target selection/test (with group fractions approximate). Keep the seed and split manifest fixed for the study.

### Synthetic installation check

The generated images are geometric shapes, **never medical evaluation data**. After installing dependencies:

```bash
python -m scripts.make_demo_data
python train.py --config config.yaml --data-dir data/demo --run-dir runs/demo_unet --epochs 2
python train.py --config config_small.yaml --data-dir data/demo --run-dir runs/demo_small --epochs 2
python select_model.py --runs runs/demo_unet runs/demo_small --target-images data/demo/target_select/images --output runs/demo_selection.json
python evaluate.py --selection runs/demo_selection.json --data-dir data/demo --split target_test
```

### Current single-seed prototype

These commands exercise the existing implementation. They do not yet run all augmentations, seeds, confidence ablations or metrics in [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Freeze the protocol before evaluating real target-test labels.

```bash
python train.py --config config.yaml
python select_model.py --runs runs/unet --target-images data/polyp/target_select/images --output runs/selection.json
python evaluate.py --selection runs/selection.json --split target_test --output runs/selected_target.json
python evaluate.py --checkpoint runs/unet/best_source.pt --split target_test --output runs/source_baseline_target.json
```

For an optional architecture comparison, train `config_small.yaml` separately and include its run directory in `select_model.py --runs`. Select the source-only comparator from the same candidate pool by its highest `source_val_dice` in `runs/selection.csv` **before** opening target results. Record selected architecture, epoch, source Dice, target stability, target Dice/IoU and random seed. Run multiple independent seeds for a research report; the commands above use only seed 42.

## Files

| File | Role |
| --- | --- |
| `models/unet.py` | Standard (4 downsamplings) and small (3) U-Net |
| `datasets/segmentation.py` | Labeled image/mask loader; image-only target loader |
| `scripts/prepare_polyp_data.py` | Deterministic source and target staging, optional group split |
| `train.py` | Source training, all epoch checkpoints, validation log, resume |
| `select_model.py` | Target image-only stability score + source validation gate |
| `evaluate.py` | Separate target test evaluation |
| `scripts/make_demo_data.py` | Synthetic installation check |

## Limitations

This first version is **source-only domain generalization plus unsupervised checkpoint selection**, not UDA training: the target images influence only the final checkpoint choice. Stability can prefer low quality masks. The target test must remain independent of selection; a group-aware split is essential for video frames. Source images may also have unknown patient overlap. Report uncertainty across seeds, inspect failure cases, and do not describe this research prototype as clinical software.

Dataset references: Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020; Bernal et al., *WM-DOVA maps for accurate polyp highlighting in colonoscopy*, Computerized Medical Imaging and Graphics 2015; Xiong et al., *Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging*, 2026.
