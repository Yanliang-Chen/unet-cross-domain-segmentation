# Experiment Proposal: Polyp Segmentation under Domain Shift with a Fixed U-Net

Status: 2026-11-01. Protocol: [benchmark_protocol.yaml](benchmark_protocol.yaml). Result tables: [results/benchmark/](results/benchmark/). Daily log: [每日进度.md](每日进度.md).

## 1. Research question

A U-Net trained on one hospital's colonoscopy images loses about 0.2 Dice on images from other hospitals and devices. The project asks:

> **Can unsupervised domain adaptation, trained only with unlabelled target images, make the original U-Net segment polyps reliably better in new domains than simply training it longer — and can the improved model be chosen without target labels?**

The network is fixed: the U-Net of Ronneberger et al., base width 16, depth 4, 256×256 input. Methods may change only the training loss or the pipeline. Discriminators and EMA teachers are used during training only, so the deployed model is always the plain U-Net.

## 2. Hypotheses

| | Hypothesis | Test |
|---|---|---|
| H1 | Some adaptation methods beat a matched continued-training control. | Paired gain over the control, with a 95% bootstrap interval, on ClinicDB |
| H2 | Segmentation-specific methods (MinEnt+MMD, DACS cross-domain mixing) add to the best single methods. | Same paired test |
| H3 | The gains hold on other targets without more false positives on normal images. | ETIS, ColonDB, PolypGen, FAR; Dice, small-polyp Dice, normal-image FPR |
| H4 (supporting) | Label-free validators become unreliable once adapted models enter the candidate pool; source validation stays the safest rule. | Selection gap and rank correlation of 19 validators, including CMR (Talks et al.) |

## 3. Setup

**Data.**

| Role | Dataset | Images | Setting |
|---|---|---|---|
| Source (labelled) | Kvasir-SEG | 800 train / 200 validation | – |
| Target 1 | CVC-ClinicDB | 185 unlabelled for adaptation / 427 labelled test, 20 video groups | inductive |
| Target 2 | ETIS-LaribPolypDB | 196 | transductive (same unlabelled images used for adaptation and test) |
| Target 3 | CVC-ColonDB | 380 | transductive |
| Target 4 | PolypGen C1–C6 | 1,537 polyp + 1,755 normal | transductive |
| Target 5 | FAR | 432 polyp + 7,749 healthy | transductive |

Target masks are never used for training or checkpoint selection.

**Source models.** 6 runs (augmentation a0/a1 × seeds 42/43/44) × 30 epochs = 180 checkpoints. Final source validation Dice is 0.78–0.83, in line with published U-Net results on Kvasir (about 0.82).

**Adaptation training.** Every method starts from a run's epoch-30 checkpoint and trains for 2,000 steps, with batches of 8 source and 8 target images, learning rate 1e-4. The source BCE+Dice loss is kept, a checkpoint is saved every 200 steps, and λ ∈ {0.1, 0.5, 1}.

**Methods.**

| Group | Methods |
|---|---|
| Control | **Continue**: same start, steps and joint source+target forward pass, no target loss (λ=0) |
| Normalisation | AdaBN |
| Output level | MinEnt, AdvEnt, pixel BNM, pixel MCC |
| Feature level | DANN (bottleneck GRL), MMD (bottleneck multi-kernel) |
| Input level | FDA (low-frequency amplitude swap) |
| Teacher | Mean Teacher (EMA 0.99) |
| New, segmentation-specific | **MinEnt+MMD** (output + feature alignment); **DACS** (source polyps pasted onto target images, EMA-teacher pseudo-labels weighted by confidence, colour jitter) |

**Deployment rule, fixed in advance.** Each trajectory (method × base run × λ) deploys the checkpoint with the best source validation Dice (Source-Risk). Each deployed model is compared with the same base run's control and with its SourceOnly pick (the source checkpoint chosen by the same rule).

**Metrics.** All metrics are computed at native resolution: probabilities are upsampled bilinearly, then thresholded at 0.5. They are:
- mean per-image Dice and IoU;
- group-mean Dice;
- small-polyp Dice (GT area < 2%);
- normal-image false-positive rate (any pixel, and area ≥ 0.1%);
- image-level AUROC.

Intervals come from 2,000 bootstrap draws over base runs and image groups jointly.

## 4. Completed

### 4.1 Infrastructure

- `benchmark/` holds the scenario caches, native-resolution scoring, the 11 adaptation methods and the control, 19 label-free validators, the selection analysis, and paired reports (`benchmark/report.py`).
- `bash benchmark/run_scenario.sh <scenario> <methods...>` runs one scenario end to end.

### 4.2 Source-only baseline on all 6 scenarios (180 checkpoints)

| Target | Best checkpoint Dice (Oracle) | Source-Risk pick | Gap |
|---|---:|---:|---:|
| ClinicDB | 0.684 | 0.684 | 0.000 |
| ColonDB | 0.534 | 0.500 | 0.034 |
| ETIS | 0.423 | 0.374 | 0.050 |
| PolypGen | 0.596 | 0.596 | 0.000 |
| FAR | 0.533 | 0.521 | 0.012 |

ColonDB and ETIS match published U-Net results for these unseen datasets (about 0.51 and 0.40 in the PraNet paper). The drop from the source domain is domain shift, not a weak baseline.

### 4.3 ClinicDB full grid (2,220 candidates): H1 and H2 supported

Each trajectory deploys its Source-Risk pick. Gains are group-mean Dice differences (from [results/benchmark/clinicdb/paired.md](results/benchmark/clinicdb/paired.md)).

| Method | Deployed Dice | Gain vs control [95% CI] | Better than control | Gain vs SourceOnly [95% CI] |
|---|---:|---|---:|---|
| **DACS (new)** | **0.686** | **+0.050 [+0.020, +0.082]** | **18/18** | +0.082 [+0.034, +0.138] |
| **MinEnt+MMD (new)** | 0.670 | +0.036 [+0.003, +0.077] | 17/18 | +0.068 [+0.020, +0.126] |
| MinEnt | 0.664 | +0.026 [−0.016, +0.069] | 14/18 | +0.058 [+0.017, +0.109] |
| MMD | 0.658 | +0.025 [+0.001, +0.052] | **18/18** | +0.056 [+0.004, +0.116] |
| DANN | 0.649 | +0.015 [+0.001, +0.031] | 14/18 | +0.046 [−0.006, +0.108] |
| FDA | 0.650 | +0.011 [−0.037, +0.061] | 13/18 | +0.042 [−0.007, +0.094] |
| Continue (control) | 0.637 | 0 | – | +0.031 [−0.027, +0.096] |
| AdvEnt | 0.635 | −0.001 [−0.012, +0.010] | 9/18 | +0.030 |
| BNM | 0.627 | −0.011 [−0.022, −0.002] | 5/18 | +0.020 |
| SourceOnly | 0.610 | −0.031 | – | 0 |
| AdaBN | 0.606 | −0.023 | 1/6 | +0.008 |
| Mean Teacher | 0.591 | −0.047 | 5/18 | −0.016 |
| MCC | 0.476 | −0.156 [−0.212, −0.104] | 0/18 | −0.125 |

- **Training longer alone helps** (+0.031 over SourceOnly). Without the control, about half of every method's gain would be misattributed to adaptation.
- **The new segmentation-specific methods are the strongest (H2).** DACS beats the control in all 18 trajectories (+0.050, interval above zero) and improves with λ: deployed Dice 0.669 / 0.690 / 0.699 at λ 0.1 / 0.5 / 1. Its small-polyp Dice is 0.481 at λ=1, against 0.393 for the control. MinEnt+MMD (+0.036, 17/18) beats either of its parts alone.
- **Among the earlier methods, only MMD and DANN have intervals against the control that exclude zero.** MinEnt has a larger mean gain but more variance.
- **The best checkpoint overall is now DACS** (Oracle 0.727). Source-Risk still picks a control checkpoint over the combined pool (0.676), so per-method deployment with a fixed rule beats global label-free selection.
- **Negative results:** MCC, Mean Teacher and AdaBN fall below the control, and BNM is slightly below it.

### 4.4 Label-free selection on ClinicDB (H4 supported)

- **Source-Risk is the best single validator.** Across the whole pool it picks 0.676 against an Oracle of 0.714, a gap of 0.038 [−0.023, 0.065]. The validator ensemble reaches 0.688.
- **Confidence-based validators fail.** Entropy, InfoMax, BNM, MCC, IWCV and MixVal-fg pick confident-but-wrong MCC models at 0.41–0.45, gaps of 0.26–0.31. Their best rank correlation over the pool is 0.45, against 0.81–0.96 within source-only runs.
- **CMR (Talks et al.):**
  - The two-class CMR picks near-empty source checkpoints from epochs 1–2 (Dice 0.12–0.15), and the collapse guard fixes this.
  - CMR-EI and the foreground-only versions pick 0.64–0.66, which does not beat Source-Risk.
  - On targets with normal images, foreground-only CMR ranks models in the wrong order (PolypGen ρ −0.23, FAR −0.54; source-only pool).
- **Labelling 4% of the target (17 images)** gives a gap of 0.019. Smaller budgets do worse than Source-Risk.
- **Scope:** all candidates share one architecture, so these results concern choosing checkpoints and methods for a fixed U-Net, not choosing between architectures.

## 5. In progress (2026-11-01)

| Job | Content | Expected |
|---|---|---|
| ETIS, reduced grid | MinEnt, MMD, DANN, FDA, MinEnt+MMD, DACS + control (H3) | tonight |

## 6. Remaining work

| # | Task | Purpose | Compute | When |
|---|---|---|---|---|
| 1 | ~~Paired report for MinEnt+MMD and DACS on ClinicDB~~ (done, see 4.3) | H2 | – | Nov 1 |
| 2 | ETIS results and paired report | H3, hardest target | queued | Nov 2 |
| 3 | ColonDB, same reduced grid | H3 | ≈ 5 h GPU | Nov 2 |
| 4 | PolypGen and FAR. Grid reduced to the best 2–3 methods, λ=0.5, 5 checkpoints; written into the protocol before running | H3, false positives on normal images | ≈ 6–8 h GPU | Nov 3 |
| 5 | Small-component post-processing (nnU-Net style, minimum area fixed in advance) on the deployed models | Cuts false positives on normal images | minutes per model | Nov 3 |
| 6 | λ sensitivity and training-step curves for the best methods; small-polyp Dice per method | Analysis | CPU | Nov 4 |
| 7 | Qualitative figures: predictions before and after adaptation on each target, including failures | Report | CPU | Nov 4 |
| 8 | Optional: source-side style augmentation in the spirit of ConStyX, only if time allows (needs 6 new source runs) | Better start point | ≈ 1 day GPU | optional |
| 9 | Write-up: method, results, limitations; the earlier 18 attempts move to an appendix ([失败方法记录.md](失败方法记录.md)) | Report | – | Nov 5–7 |

**Decision rules fixed now.**
- A method counts as helpful on a target only if its paired gain over the control is positive in at least 2/3 of trajectories, and its interval excludes zero or it does so on at least two targets.
- A method that raises the normal-image false-positive rate by more than 5 points on PolypGen or FAR is reported as a trade-off, not a success.

## 7. Limitations and risks

- **Few independent runs.** The real sample size is 6 base runs, since λ values within a run are correlated. Intervals are wide; MinEnt's interval against the control still includes zero.
- **Transductive targets.** ETIS, ColonDB, PolypGen and FAR use the same images for adaptation and test. That is standard in UDA, but those results are optimistic compared with ClinicDB's held-out test split.
- **Colonoscopy only.** "Domains" means hospitals, devices and cohorts within colonoscopy, not other imaging modalities.
- **Strict evaluation.** Small model, native resolution, and plain per-image Dice. Absolute numbers are lower than in papers that train on ClinicDB or evaluate at 352×352; conclusions rest on paired gains, not absolute Dice.
- **Not done on purpose:** architecture comparisons (UNet++, nnU-Net), which would change the fixed network and the research question.
