# Robust Cross-Domain Biomedical Image Segmentation with U-Net and Label-Free Model Selection

A small, reproducible research starting point for **binary polyp segmentation**. Train a standard U-Net on labeled Kvasir-SEG images, use **unlabeled CVC-ClinicDB images** to choose one candidate checkpoint, and only then evaluate on held-out CVC-ClinicDB masks.

The proposed full experiment, including how the segmentation task relates to Xiong et al.'s classification datasets, is in [实验.md](实验.md). It distinguishes planned analyses from features already implemented.

For the current status and cross-machine handoff, start with [提交进度记录.md](提交进度记录.md).
The completed six-run real-data results are in [results/2026-10-26-real-polyp.md](results/2026-10-26-real-polyp.md).

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

On macOS (Python 3.11 and PyTorch 2.11.0, matching the Windows PyTorch release):

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
python -m scripts.check_device
```

The Mac wheel does not include CUDA. MPS availability depends on the local OS and hardware;
the setup script reports it but does not require it. CPU is sufficient for workflow checks.

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
python -m scripts.prepare_polyp_data --source data/raw/kvasir-seg --target data/raw/cvc-clinicdb --target-groups-csv data/raw/cvc-clinicdb/metadata.csv --require-target-groups
```

If source patient identifiers are available, also pass `--source-groups-csv <CSV>` using the same `sample_id,group_id` format. If you lack reliable target sequence identifiers, omit both `--target-groups-csv` and `--require-target-groups` for an **exploratory** image-level split only; adjacent video frames may appear in both target partitions and overstate generalization. Never infer video IDs from consecutive filenames without source documentation.

The preparation script checks image/mask pairing, dimensions and nonempty masks, rejects exact duplicate images crossing splits, and stages output atomically. It writes `split_manifest.csv` with split, sample ID, group ID and image SHA-256. It refuses to overwrite populated output. By default: 80/20 source train/validation, 30/70 target selection/test (with group fractions approximate). Keep the seed and split manifest fixed for the study.

### Synthetic installation check

The generated images are geometric shapes, **never medical evaluation data**. After installing dependencies:

```bash
python -m scripts.make_demo_data
python train.py --config config.yaml --data-dir data/demo --run-dir runs/demo_unet --epochs 2
python select_model.py --runs runs/demo_unet --target-images data/demo/target_select/images --output runs/demo_selection.json
python evaluate.py --selection runs/demo_selection.json --data-dir data/demo --split target_test --final-evaluation
```

### Frozen real-data experiment

On the Windows data machine, use the audited PNG staging directory `data/polyp_png`.
The TIFF staging directory is not readable by the current Pillow installation. Before
training, validate the staged files and print the manifest hash:

```bash
python -m scripts.check_protocol --show-unlocked-hash
```

Put the printed SHA-256 in `protocol.yaml` under `split_manifest_sha256`, commit the
protocol, then run `python -m scripts.check_protocol` again. This checks counts,
sequence isolation, image/mask IDs, every staged image hash and the absence of
`target_select/masks`. The hash cannot be filled on a machine without the private
dataset. Keep the manifest and all medical data out of Git.

The matrix contains A0 (horizontal/vertical flips) and A1 (the same flips plus mild
brightness, contrast and RGB channel gains), each with seeds 42, 43 and 44. Each run
has 30 epochs. Run one pilot cell first to check memory and timing, then the matrix:

```bash
python -m scripts.run_matrix --dry-run --device cuda
python -m scripts.run_matrix --only a0_seed42 --device cuda
python -m scripts.run_matrix --device cuda
```

The runner safely resumes an interrupted run from `latest.pt` and skips cells that
already have a selection file. It saves resolved configs, environment metadata,
per-epoch checkpoints, source metrics and one target image-only selection file per
run. The source, stability and entropy choices are fixed within each run's candidate
pool; the source Dice guardrail is 0.03. Mean foreground area and empty-prediction
fraction diagnose trivial masks. No target-test labels are read in this stage.

After all six training and selection files exist and the protocol is frozen, run:

```bash
python -m scripts.final_analysis --device cuda
python -m scripts.posthoc_diagnostics
```

This explicitly unlocks final target-test evaluation. It checks the six selection
files and their protocol/data hashes before reading target-test labels, then reports
per-image Dice, IoU, Precision and Recall, the paired stability-minus-source Dice
difference, video-sequence bootstrap intervals, seed summaries and a **post-hoc**
Oracle upper bound. Repeating the command with the same frozen inputs resumes an
interrupted final analysis. It does not revise the selection rule based on test results.

## Files

| File | Role |
| --- | --- |
| `models/unet.py` | Standard U-Net with four downsamplings |
| `datasets/segmentation.py` | Labeled image/mask loader; image-only target loader |
| `scripts/prepare_polyp_data.py` | Deterministic source and target staging, optional group split |
| `train.py` | Source training, all epoch checkpoints, validation log, resume |
| `select_model.py` | Target image-only stability score + source validation gate |
| `evaluate.py` | Separate target test evaluation |
| `protocol.yaml` | Frozen design and split-manifest lock |
| `scripts/check_protocol.py` | Validate the locked split and staged image hashes |
| `scripts/run_matrix.py` | Six-run training and image-only selection runner |
| `scripts/final_analysis.py` | Frozen final evaluation and paired report |
| `scripts/posthoc_diagnostics.py` | Post-hoc candidate rank and failure diagnostics |
| `scripts/make_demo_data.py` | Synthetic installation check |

## Limitations

This first version is **source-only domain generalization plus unsupervised checkpoint selection**, not UDA training: the target images influence only the final checkpoint choice. Stability can prefer low quality masks. The target test must remain independent of selection; a group-aware split is essential for video frames. Source images may also have unknown patient overlap. Report uncertainty across seeds, inspect failure cases, and do not describe this research prototype as clinical software.

Dataset references: Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MMM 2020; Bernal et al., *WM-DOVA maps for accurate polyp highlighting in colonoscopy*, Computerized Medical Imaging and Graphics 2015; Xiong et al., *Towards Practical Algorithm Selection for Unsupervised Domain Adaptation in Medical Imaging*, 2026.
