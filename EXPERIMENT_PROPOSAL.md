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

### 4.5 ETIS reduced grid (1,500 candidates): H3 supported on the hardest target

Transductive setting, bootstrap over images ([results/benchmark/etis/paired.md](results/benchmark/etis/paired.md)).

| Method | Deployed Dice | Gain vs control [95% CI] | Better than control | Gain vs SourceOnly |
|---|---:|---|---:|---:|
| FDA | 0.430 | +0.092 [+0.058, +0.128] | 18/18 | +0.064 |
| MinEnt+MMD | 0.412 | +0.074 [+0.034, +0.114] | 16/18 | +0.046 |
| DACS | 0.403 | +0.066 [+0.032, +0.101] | 17/18 | +0.038 |
| MinEnt | 0.393 | +0.055 [+0.021, +0.089] | 18/18 | +0.027 |
| MMD | 0.388 | +0.050 [+0.018, +0.081] | 16/18 | +0.022 |
| DANN | 0.362 | +0.024 [+0.007, +0.042] | 15/18 | −0.004 |
| SourceOnly | 0.366 | – | – | 0 |
| Continue (control) | 0.338 | 0 | – | −0.028 |

- **On ETIS, training longer hurts** (−0.028 against SourceOnly), so every adaptation method beats the control clearly.
- **The best method depends on the target.** FDA is best on ETIS but only mid-ranked on ClinicDB, and DACS is the reverse. **MinEnt+MMD is second on both targets**, which makes it the most consistent method so far.

### 4.6 ColonDB reduced grid (1,500 candidates): H3 supported

Transductive setting, 380 images without video IDs, so the bootstrap is over images ([results/benchmark/colondb/paired.md](results/benchmark/colondb/paired.md)).

| Method | Deployed Dice | Gain vs control [95% CI] | Better than control | Gain vs SourceOnly [95% CI] |
|---|---:|---|---:|---|
| DACS | 0.609 | +0.174 [+0.130, +0.223] | 18/18 | +0.165 [+0.135, +0.197] |
| MinEnt+MMD | 0.568 | +0.133 [+0.079, +0.192] | 18/18 | +0.124 [+0.086, +0.168] |
| MMD | 0.541 | +0.106 [+0.062, +0.158] | 18/18 | +0.097 [+0.058, +0.141] |
| MinEnt | 0.534 | +0.098 [+0.038, +0.166] | 18/18 | +0.089 [+0.049, +0.130] |
| FDA | 0.508 | +0.073 [+0.021, +0.131] | 15/18 | +0.064 [+0.029, +0.101] |
| DANN | 0.498 | +0.062 [+0.029, +0.102] | 17/18 | +0.053 [+0.001, +0.101] |
| AdaBN | 0.458 | +0.023 [−0.013, +0.063] | 4/6 | +0.014 |
| SourceOnly | 0.444 | – | – | 0 |
| Continue (control) | 0.435 | 0 | – | −0.009 |

- **The largest gains so far.** Every method's interval excludes zero against both the control and SourceOnly; DACS adds +0.165 Dice over SourceOnly.
- **Ranking across targets.** DACS is first on ClinicDB and ColonDB but third on ETIS; FDA is first on ETIS but fifth here. MinEnt+MMD is in the top two on all three targets and remains the most consistent method.
- **Selection matters more here.** DACS reaches an Oracle of 0.711, but per-trajectory Source-Risk picks average 0.609; global Source-Risk on the merged pool picks a control checkpoint (0.488, gap 0.224). Fixed per-method deployment is again better than global label-free selection.

**Summary across the three targets (gain over SourceOnly, interval excludes zero):** DACS and MinEnt+MMD on 3/3; FDA on 2/3 (ETIS, ColonDB; not ClinicDB); MinEnt on 2/3 (borderline on ETIS); MMD on 2/3; DANN on 1/3.

### 4.7 FAR reduced grid (λ=0.5, checkpoints every 400 steps, 6 trajectories per method)

Adaptation images include the 7,749 healthy images. FAR was used for method development before 2026-11-01. Normal FPR = share of healthy images with any predicted polyp pixel ([results/benchmark/far/paired.md](results/benchmark/far/paired.md)).

| Method | Deployed Dice | Normal FPR | Gain vs control [95% CI] | Better than control | Gain vs SourceOnly [95% CI] |
|---|---:|---:|---|---:|---|
| DACS | 0.565 | **45.0%** | +0.079 [+0.045, +0.124] | 6/6 | +0.063 [+0.036, +0.092] |
| FDA | 0.563 | 54.4% | +0.066 [+0.024, +0.116] | 6/6 | +0.050 [+0.020, +0.084] |
| MinEnt+MMD | 0.542 | 65.2% | +0.054 [+0.021, +0.095] | 6/6 | +0.038 [+0.011, +0.065] |
| SourceOnly | 0.510 | 56.5% | – | – | 0 |
| Continue (control) | 0.492 | 73.7% | 0 | – | −0.016 |
| AdaBN | 0.426 | 90.6% | −0.065 | 0/6 | −0.081 |

- All three methods beat both references in 6/6 trajectories, with intervals excluding zero.
- **DACS improves Dice and lowers normal-image FPR by 11.5 points.** FDA keeps FPR at the SourceOnly level. MinEnt+MMD raises it by 8.7 points, which the decision rule classes as a trade-off.
- Continued training and AdaBN raise FPR sharply (73.7%, 90.6%).
- Global Source-Risk on the merged pool picks a control checkpoint (0.539, FPR 65.6%; Oracle 0.592, FDA).

### 4.8 PolypGen (λ 0.1/0.5/1, checkpoints every 400 steps, 18 trajectories per method)

Adaptation images include the 1,755 normal images; bootstrap over centre/sequence groups. λ=0.5 was pre-specified; λ 0.1 and 1 were added after seeing it (recorded in the protocol). The λ=0.5-only table, [paired_lam0.5.md](results/benchmark/polypgen/paired_lam0.5.md), leads to the same conclusions.

| Method | Deployed Dice | Normal FPR | Gain vs control [95% CI] | Better than control | Gain vs SourceOnly [95% CI] |
|---|---:|---:|---|---:|---|
| FDA | 0.611 | 37.1% | +0.025 [−0.003, +0.058] | 14/18 | +0.037 [+0.015, +0.061] |
| MinEnt+MMD | 0.603 | 39.7% | +0.014 [−0.010, +0.046] | 12/18 | +0.026 [+0.011, +0.043] |
| DACS | 0.599 | 36.8% | +0.011 [−0.007, +0.030] | 12/18 | +0.023 [−0.009, +0.048] |
| Continue (control) | 0.592 | 79.1% | 0 | – | +0.012 |
| SourceOnly | 0.573 | 24.7% | −0.012 | – | 0 |
| AdaBN | 0.469 | 82.2% | −0.106 | 0/6 | −0.094 |

- **Smallest gains of all targets.** Small gains over SourceOnly; all intervals against the control include zero. PolypGen starts from a higher baseline and is closer to the source.
- **Continued training raises normal-image FPR from 24.7% to 79.1%**; all three methods remove most of that.
- **λ controls false positives.** DACS at λ 0.1/0.5/1: FPR 60.2%/25.5%/24.7%, Dice 0.590/0.601/0.606, so at λ≥0.5 DACS matches SourceOnly's FPR with higher Dice. MinEnt+MMD behaves the same way (52.1%/34.0%/32.9%); FDA does not (35.3%–39.7%).

**Summary over five targets (gain over SourceOnly, interval excludes zero):** MinEnt+MMD 5/5; DACS 4/5 (not PolypGen); FDA 4/5 (not ClinicDB). DACS is first on ClinicDB, ColonDB and FAR; MinEnt+MMD is in the top three on every target.

### 4.9 Small-component post-processing (all five targets)

Rule fixed in the protocol before any post-processed prediction: remove 8-connected predicted components below 0.1% of the native image (primary; the existing area-FPR threshold) or 0.41% (sensitivity; smallest main polyp component in the 800 source training masks). Applied to every deployed pick of SourceOnly, control, DACS, MinEnt+MMD and FDA. Helpful if normal FPR falls by at least 5 points and Dice by at most 0.01 on PolypGen and FAR. Table: `results/benchmark/postprocessing.md`.

| Method | PolypGen FPR raw → 0.1% → 0.41% | PolypGen ΔDice (0.1%) | FAR FPR raw → 0.1% → 0.41% | FAR ΔDice (0.1%) |
|---|---|---:|---|---:|
| DACS | 36.8% → 32.2% → 27.8% | +0.015 | 45.0% → 38.0% → 31.4% | +0.000 |
| MinEnt+MMD | 39.7% → 35.1% → 29.9% | +0.005 | 65.2% → 58.7% → 51.0% | +0.000 |
| FDA | 37.1% → 32.2% → 26.6% | +0.005 | 54.4% → 48.2% → 41.3% | +0.000 |
| Control | 79.1% → 73.9% → 67.3% | +0.008 | 73.7% → 68.5% → 62.7% | +0.001 |
| SourceOnly | 24.7% → 21.4% → 17.7% | +0.006 | 56.5% → 51.3% → 45.0% | −0.000 |

- **Primary threshold narrowly misses the pre-set rule.** FAR: FPR falls 5.1–6.9 points (group-weighted, all intervals exclude zero), which passes. PolypGen: the three methods fall only 4.4–4.8 points, just under 5.
- **No Dice cost.** Dice changes by at most ±0.001 on ClinicDB, ETIS, ColonDB and FAR. On PolypGen it rises by 0.005–0.015, mostly on small polyps (DACS small-polyp Dice 0.364 → 0.415), so small blobs there are mostly false detections.
- **Sensitivity threshold (0.41%)** cuts FPR by 6.9–14.0 points on both targets with Dice change −0.007 to +0.019 and would pass. Switching to it after seeing results would be post hoc, so it is reported as sensitivity only.
- **Post-processing does not change the ranking or replace adaptation.** The control still flags 68–74% of normal images; most false positives are components larger than 0.1% of the image.

### 4.10 λ, training length and small polyps (CPU, from the tracked candidate tables)

Script `benchmark/curves.py`; table `results/benchmark/curves.md`; figures `results/benchmark/figures/{step_curves,step_fpr,lambda}.png`.

**λ.**
- DACS improves monotonically with λ on all four targets with three λ values. At λ 0.1 / 1:
  - ClinicDB 0.669 / 0.699
  - ETIS 0.374 / 0.422
  - ColonDB 0.545 / 0.658
  - PolypGen 0.590 / 0.606, with normal FPR 60% / 25%
- FDA goes the other way: it is best at λ=0.1 on ClinicDB, ColonDB and PolypGen.
- MinEnt+MMD is flat.
- The main results average over λ. Recommending λ=1 for DACS is a post hoc observation, not a change to the deployed numbers.

**Training length.**
- All three methods pass SourceOnly within 200–400 steps.
- The control drifts below its start on ETIS, ColonDB and FAR (ColonDB 0.443 → 0.410).
- MinEnt+MMD peaks early: the mean pick is at steps 480–760, and the last step is clearly worse (ColonDB 0.496 vs 0.568 at the pick), so checkpoint selection matters for it.
- DACS keeps improving on ColonDB.
- The Source-Risk pick is close to the best checkpoint in each trajectory: the gap is 0.003–0.032 for the methods and 0.008–0.069 for the control.
- Over training, the control's normal FPR climbs to 83–85%, while DACS on FAR falls to about 41%.

**Small polyps (GT under 2%).** Gain over the same base run's SourceOnly pick, in trajectories won:

| Target | DACS | MinEnt+MMD | FDA |
|---|---|---|---|
| ETIS | +0.036 (14/18) | +0.037 (13/18) | +0.077 (18/18) |
| ColonDB | +0.116 (16/18) | +0.083 (16/18) | +0.087 (18/18) |
| FAR | +0.063 (6/6) | +0.031 (5/6) | +0.045 (6/6) |

- On ClinicDB and PolypGen the small-polyp gains are close to zero (+0.005 to +0.023), so the overall gains there come from larger polyps.
- The control (−0.017 to −0.063) and AdaBN lower small-polyp Dice on every target.

### 4.11 Qualitative examples

`benchmark/qualitative.py` writes `results/benchmark/figures/qualitative_<target>.png` and lists the chosen images in `results/benchmark/qualitative.md`.

**What is shown.** Models from base run a0_seed42 (Source-Risk pick, λ=0.5): SourceOnly, control, DACS, MinEnt+MMD and FDA.

**How images were chosen.** By a fixed rule using those same models, not by eye:
- the polyp images where DACS gains most, at the median, and least over SourceOnly;
- a small polyp at the median DACS Dice;
- on PolypGen and FAR, two normal images: one flagged by the control but clean under DACS and SourceOnly, and one flagged by DACS but clean under SourceOnly.

**What they show.**
- **The typical gain is recovering a polyp SourceOnly missed entirely.** SourceOnly scores Dice 0 while the adapted models score 0.9 or higher, on every target.
- **The control fails with large false regions** on background mucosa, instruments or over-exposed areas, which matches its 74–79% normal FPR.
- **The adapted models fail by losing low-contrast small polyps that SourceOnly found**, as in the worst rows on ColonDB and FAR (DACS 0.00 against SourceOnly 0.71–0.83), or by spreading onto instruments and glare.
- **DACS false positives on normal images are small blobs** on over-exposed or yellow-tinted frames, the kind small-component removal targets. FDA's blobs on the same frames are larger.

### 4.12 Summary figures

`benchmark/summary_figures.py` reads only tables already in git.

- **`figures/gains.png`:** paired gain with 95% interval over SourceOnly and over the control, for each method on each target. An open marker means the interval includes zero.
  - Over the control, every method is positive on ClinicDB, ETIS, ColonDB and FAR. ClinicDB FDA is the one exception, with an interval that includes zero.
  - PolypGen is the only target where all three methods' intervals include zero.
  - The control's interval against SourceOnly includes zero on every target.
- **`figures/dice_fpr.png`:** deployed Dice against normal-image FPR on PolypGen and FAR. Small dots are single trajectories; large markers are method means.
  - FAR: DACS sits top-left, with the highest Dice and the lowest FPR.
  - On both targets the control sits far right, at 74–79% FPR.
  - PolypGen: all three methods beat SourceOnly on Dice but have 12–15 points more FPR.

## 5. In progress

Nothing running (summary figures finished 2026-11-02; only the write-up remains).

## 6. Remaining work

| # | Task | Purpose | Compute | When |
|---|---|---|---|---|
| 1 | ~~Paired report for MinEnt+MMD and DACS on ClinicDB~~ (done, see 4.3) | H2 | – | Nov 1 |
| 2 | ~~ETIS results and paired report~~ (done, see 4.5) | H3 | – | Nov 1 |
| 3 | ~~ColonDB, same reduced grid~~ (done, see 4.6) | H3 | – | Nov 1 |
| 4 | PolypGen and FAR: DACS, MinEnt+MMD, FDA + control, λ=0.5, 5 checkpoints (done, see 4.7 and 4.8; PolypGen extended to λ 0.1/1 after seeing λ=0.5) | H3, false positives on normal images | ≈ 5 h GPU | Nov 1–2 |
| 5 | ~~Small-component post-processing (nnU-Net style, minimum area fixed in advance) on the deployed models~~ (done, see 4.9) | Cuts false positives on normal images | ≈ 2 h GPU inference | Nov 2 |
| 6 | ~~λ sensitivity and training-step curves for the best methods; small-polyp Dice per method~~ (done, see 4.10) | Analysis | CPU | Nov 2 |
| 7 | ~~Qualitative figures: predictions before and after adaptation on each target, including failures~~ (done, see 4.11) | Report | GPU inference, minutes | Nov 2 |
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
