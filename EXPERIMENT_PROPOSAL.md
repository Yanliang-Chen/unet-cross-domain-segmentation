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
| H2 | Segmentation-specific methods (MinEnt+MMD, DACS cross-domain mixing) add to the best single methods. | Direct paired comparison with named single methods (amendment v2): MinEnt+MMD vs MinEnt, MMD; DACS vs MinEnt, MMD, FDA; ClinicDB, ETIS, ColonDB |
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
| Target 4 | PolypGen C1–C6 | 1,412 polyp + 125 empty-mask (polyp set, no polyp) + 1,755 normal | transductive |
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
- mean per-image Dice and IoU on images with a non-empty mask;
- group-mean Dice;
- small-polyp Dice (0 < GT area < 2%);
- empty-mask FPR: polyp-set images whose mask is empty (PolypGen only) with any predicted pixel;
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
| PolypGen | 0.616 | 0.616 | 0.000 |
| FAR | 0.533 | 0.521 | 0.012 |

ColonDB and ETIS match published U-Net results for these unseen datasets (about 0.51 and 0.40 in the PraNet paper). The drop from the source domain is domain shift, not a weak baseline.

### 4.3 ClinicDB full grid (2,220 candidates): H1 supported; H2 see 4.13

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
- **The new segmentation-specific methods have the highest means.** DACS beats the control in all 18 trajectories (+0.050, interval above zero) and improves with λ: deployed Dice 0.669 / 0.690 / 0.699 at λ 0.1 / 0.5 / 1. Its small-polyp Dice is 0.481 at λ=1, against 0.393 for the control. MinEnt+MMD (+0.036, 17/18) has a higher mean than either part, but the direct comparison does not exclude zero on ClinicDB (4.13).
- **Among the earlier methods, only MMD and DANN have intervals against the control that exclude zero.** MinEnt has a larger mean gain but more variance.
- **The best checkpoint overall is now DACS** (Oracle 0.727). Source-Risk still picks a control checkpoint over the combined pool (0.676), so per-method deployment with a fixed rule beats global label-free selection.
- **Negative results:** MCC, Mean Teacher and AdaBN fall below the control, and BNM is slightly below it.

### 4.4 Label-free selection (H4): v1 statements on the 1,860-candidate pool

These results come from ClinicDB's v1 pool of 1,860 candidates (before DACS and MinEnt+MMD), the only adapted pool where all 19 validators were scored. The current pools are covered by the unified table in 4.13.

- **Source-Risk was the best single validator** on that pool: 0.676 against an Oracle of 0.714, a gap of 0.038 [−0.023, 0.065].
- **Confidence-based validators fail.** Entropy, InfoMax, BNM, MCC, IWCV and MixVal-fg pick confident-but-wrong MCC models at 0.41–0.45.
- **CMR (Talks et al.):** the two-class version picks near-empty checkpoints from epochs 1–2 (Dice 0.12–0.15) unless the collapse guard is applied; foreground-only CMR ranks models in the wrong order on targets with normal images (PolypGen ρ −0.23, FAR −0.54; source-only pool).
- **Labelling 4% of the target (17 images)** gave a gap of 0.019 on that pool.
- **Scope:** all candidates share one architecture, so these results concern choosing checkpoints and methods for a fixed U-Net.

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
| FDA | 0.629 | 37.1% | +0.021 [−0.005, +0.048] | 13/18 | +0.048 [+0.021, +0.074] |
| DACS | 0.617 | 36.8% | +0.005 [−0.014, +0.024] | 12/18 | +0.032 [+0.014, +0.052] |
| MinEnt+MMD | 0.615 | 39.7% | +0.002 [−0.016, +0.020] | 9/18 | +0.029 [+0.014, +0.045] |
| Continue (control) | 0.615 | 79.1% | 0 | – | +0.027 [+0.009, +0.044] |
| SourceOnly | 0.582 | 24.7% | −0.027 | – | 0 |
| AdaBN | 0.504 | 82.2% | −0.086 | 0/6 | −0.059 |

v2 metrics (images with a non-empty mask; see 4.13). v1 numbers, which counted the 125 empty-mask images, are in `results/benchmark/metric_correction.md`.

- **No method beats the control.** All three methods and the control beat SourceOnly (intervals above zero), but every interval against the control includes zero: on PolypGen the gain comes from training longer with target images in BatchNorm, not from the adaptation loss. PolypGen starts from a higher baseline and is closer to the source.
- **Continued training raises normal-image FPR from 24.7% to 79.1%**; all three methods remove most of that.
- **λ controls false positives.** DACS at λ 0.1/0.5/1: FPR 60.2%/25.5%/24.7%, Dice 0.610/0.618/0.625, so at λ≥0.5 DACS matches SourceOnly's FPR with higher Dice. MinEnt+MMD behaves the same way (52.1%/34.0%/32.9%); FDA does not (35.3%–39.7%).

**Summary over five targets (gain over SourceOnly, interval excludes zero, v2 metrics):** DACS 5/5; MinEnt+MMD 5/5; FDA 4/5 (not ClinicDB). Against the control, all three exclude zero on ETIS, ColonDB and FAR, DACS and MinEnt+MMD also on ClinicDB, and none on PolypGen. DACS is first on ClinicDB, ColonDB and FAR; MinEnt+MMD is in the top three on every target.

### 4.9 Small-component post-processing (all five targets)

Rule fixed in the protocol before any post-processed prediction: remove 8-connected predicted components below 0.1% of the native image (primary; the existing area-FPR threshold) or 0.41% (sensitivity; smallest main polyp component in the 800 source training masks). Applied to every deployed pick of SourceOnly, control, DACS, MinEnt+MMD and FDA. Helpful if normal FPR falls by at least 5 points and Dice by at most 0.01 on PolypGen and FAR. Table: `results/benchmark/postprocessing.md`.

| Method | PolypGen FPR raw → 0.1% → 0.41% | PolypGen ΔDice (0.1%) | FAR FPR raw → 0.1% → 0.41% | FAR ΔDice (0.1%) |
|---|---|---:|---|---:|
| DACS | 36.8% → 32.2% → 27.8% | −0.000 | 45.0% → 38.0% → 31.4% | +0.000 |
| MinEnt+MMD | 39.7% → 35.1% → 29.9% | −0.000 | 65.2% → 58.7% → 51.0% | +0.000 |
| FDA | 37.1% → 32.2% → 26.6% | −0.000 | 54.4% → 48.2% → 41.3% | +0.000 |
| Control | 79.1% → 73.9% → 67.3% | +0.000 | 73.7% → 68.5% → 62.7% | +0.001 |
| SourceOnly | 24.7% → 21.4% → 17.7% | −0.001 | 56.5% → 51.3% → 45.0% | −0.000 |

- **Primary threshold narrowly misses the pre-set rule.** FAR: FPR falls 5.1–6.9 points (group-weighted, all intervals exclude zero), which passes. PolypGen: the three methods fall only 4.4–4.8 points, just under 5.
- **No Dice cost and no Dice gain.** Dice changes by at most ±0.001 on every target, small polyps included. (v1 reported a PolypGen rise of 0.005–0.015 "on small polyps"; it came from the empty-mask images, where deleting a blob turns Dice 0 into 1. See 4.13.)
- **Sensitivity threshold (0.41%)** cuts FPR by 6.9–14.0 points on both targets with Dice change −0.007 to +0.002 (v2) and would pass. Switching to it after seeing results would be post hoc, so it is reported as sensitivity only.
- **Post-processing does not change the ranking or replace adaptation.** The control still flags 68–74% of normal images; most false positives are components larger than 0.1% of the image.

### 4.10 λ, training length and small polyps (CPU, from the tracked candidate tables)

Script `benchmark/curves.py`; table `results/benchmark/curves.md`; figures `results/benchmark/figures/{step_curves,step_fpr,lambda}.png`.

**λ.**
- DACS improves monotonically with λ on all four targets with three λ values. At λ 0.1 / 1:
  - ClinicDB 0.669 / 0.699
  - ETIS 0.374 / 0.422
  - ColonDB 0.545 / 0.658
  - PolypGen 0.610 / 0.625, with normal FPR 60% / 25%
- FDA goes the other way: it is best at λ=0.1 on ClinicDB, ColonDB and PolypGen.
- MinEnt+MMD is flat.
- The main results average over λ. Recommending λ=1 for DACS is a post hoc observation, not a change to the deployed numbers.

**Training length.**
- All three methods pass SourceOnly within 200–400 steps.
- The control drifts below its start on ETIS, ColonDB and FAR (ColonDB 0.443 → 0.410).
- MinEnt+MMD peaks early: the mean pick is at steps 480–760, and the last step is clearly worse (ColonDB 0.496 vs 0.568 at the pick), so checkpoint selection matters for it.
- DACS keeps improving on ColonDB.
- The Source-Risk pick is close to the best checkpoint in each trajectory: the gap is 0.003–0.032 for the methods and 0.005–0.069 for the control.
- Over training, the control's normal FPR climbs to 83–85%, while DACS on FAR falls to about 41%.

**Small polyps (GT under 2%).** Gain over the same base run's SourceOnly pick, in trajectories won:

| Target | DACS | MinEnt+MMD | FDA |
|---|---|---|---|
| ETIS | +0.036 (14/18) | +0.037 (13/18) | +0.077 (18/18) |
| ColonDB | +0.116 (16/18) | +0.083 (16/18) | +0.087 (18/18) |
| FAR | +0.063 (6/6) | +0.031 (5/6) | +0.045 (6/6) |
| PolypGen (293 real small polyps, v2) | +0.041 (16/18) | +0.034 (14/18) | +0.036 (17/18) |

- On ClinicDB (36 small polyps) the small-polyp gains are close to zero (+0.005 to +0.021), so the overall gain there comes from larger polyps.
- The control lowers small-polyp Dice on ClinicDB, ETIS, ColonDB and FAR (−0.019 to −0.063) but raises it on PolypGen (+0.033, 5/6). AdaBN lowers it everywhere. (v1 reported near-zero PolypGen gains and a control loss there; both came from counting empty masks as small polyps.)

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
  - The control's interval against SourceOnly includes zero on every target except PolypGen (+0.027 [+0.009, +0.044]).
- **`figures/dice_fpr.png`:** deployed Dice against normal-image FPR on PolypGen and FAR. Small dots are single trajectories; large markers are method means.
  - FAR: DACS sits top-left, with the highest Dice and the lowest FPR.
  - On both targets the control sits far right, at 74–79% FPR.
  - PolypGen: all three methods beat SourceOnly on Dice but have 12–15 points more FPR.

### 4.13 Amendment v2 (2026-11-13): corrected metrics, H2, endpoints, selection, deployment

Fixed in `benchmark_protocol.yaml` (amendment_v2) before any of these numbers were computed. Scripts: `benchmark/compare.py`, `benchmark/selection.py`.

**Metric correction.**
- PolypGen's 125 empty-mask images are now excluded from Dice and small-polyp Dice and reported as a separate false-positive rate.
- Nothing changes on the other four targets.
- The changed numbers are listed in `results/benchmark/metric_correction.md`.
- What it changes on PolypGen:
  - DACS now beats SourceOnly (+0.032 [+0.014, +0.052]).
  - No method beats the control.
  - Post-processing no longer raises Dice.
  - Small-polyp gains are +0.03 to +0.04 rather than near zero.

**H2: direct paired comparisons** (`results/benchmark/h2.md`).

| Method | Comparator | ClinicDB | ETIS | ColonDB |
|---|---|---|---|---|
| MinEnt+MMD | MinEnt | +0.010 [−0.016, +0.034] | +0.019 [−0.003, +0.042] | +0.034 [+0.020, +0.049] |
| MinEnt+MMD | MMD | +0.011 [−0.006, +0.033] | +0.024 [+0.011, +0.040] | +0.027 [+0.013, +0.042] |
| DACS | MinEnt | +0.024 [−0.005, +0.049] | +0.011 [−0.011, +0.035] | +0.076 [+0.052, +0.099] |
| DACS | MMD | +0.025 [+0.003, +0.048] | +0.016 [−0.004, +0.035] | +0.068 [+0.048, +0.087] |
| DACS | FDA | +0.040 [+0.004, +0.075] | −0.027 [−0.053, −0.002] | +0.101 [+0.077, +0.126] |

- Both methods beat every named comparator only on ColonDB (1/3 targets).
- By the pre-set rule, H2 is "means support it, direct comparison inconclusive".
- DACS is significantly worse than FDA on ETIS.

**Endpoints against SourceOnly** (`results/benchmark/endpoints.md`).

| Target | DACS | MinEnt+MMD | FDA | Control |
|---|---|---|---|---|
| ClinicDB | Dice better | Dice better | not enough evidence | not enough evidence |
| ETIS | Dice better | Dice better | Dice better | not enough evidence |
| ColonDB | Dice better | Dice better | Dice better | not enough evidence |
| PolypGen | trade-off (Dice +0.032, FPR +11.2 pts) | trade-off (+0.029, +15.1 pts) | trade-off (+0.048, +10.7 pts) | trade-off (+0.027, +52.7 pts) |
| FAR | **both** (Dice +0.063, FPR −10.8 pts) | trade-off (+0.038, +8.5 pts) | Dice better (FPR −1.3 [−8.9, +5.9]) | FPR higher (+16.6 pts) |

- **Small-polyp Dice** has an interval above zero:
  - for all three methods on ColonDB and PolypGen;
  - for DACS only on FAR;
  - for FDA only on ETIS;
  - for none on ClinicDB.
- None of the three methods significantly lowers small-polyp Dice.
- **Empty-mask FPR** intervals all include zero.

**Selection over whole pools** (`results/benchmark/selection.md`).
- **Validator coverage.** Only 9 of the 19 validators are scored for every candidate in the current pools. The feature-based IWCV, DEV and SND, plus MixVal and CMR-TTD, were run only on ClinicDB's SourceOnly pool and its v1 1,860-candidate pool.
- **Source-Risk ranking.** Among those 9 validators, Source-Risk has the smallest loss on 4/5 targets and ranks 3rd on FAR, where CMR-EI does better.
- **Cross-method selection is unreliable.** Source-Risk picks a control checkpoint on ClinicDB, ColonDB and FAR. The loss is 0.051 [−0.020, 0.099] on ClinicDB (Oracle 0.727, DACS) and 0.224 on ColonDB.
- **Label budget on the current ClinicDB pool:**

  | Labelled share | Loss |
  |---|---:|
  | 1% | 0.070 |
  | 2% | 0.050 |
  | 3% | 0.053 |
  | 4% | 0.047 |
  | 5% | 0.038 |

  Only 5% beats Source-Risk.

**Deployment rule (post hoc)** (`results/benchmark/deployment.md`).

How it works:
- **Method and λ:** chosen on ClinicDB alone among {DACS, MinEnt+MMD, FDA} × {0.1, 0.5, 1}. It picks **DACS, λ=1**.
- **Checkpoint:** Source-Risk pick within each trajectory.
- **Single model:** the highest-Source-Risk pick across the 6 base runs, which is a1_seed42 on every target.

| Target | Single model Dice | Normal FPR | 6-run gain vs SourceOnly [95% CI] | SourceOnly single model Dice / FPR |
|---|---:|---:|---|---|
| ClinicDB | 0.687 | – | +0.093 [+0.045, +0.151] | 0.684 / – |
| ETIS | 0.468 | – | +0.056 [+0.025, +0.087] | 0.374 / – |
| ColonDB | 0.683 | – | +0.214 [+0.187, +0.245] | 0.500 / – |
| PolypGen | 0.631 | 16.0% | +0.040 [+0.022, +0.061] | 0.616 / 30.8% |
| FAR (λ=0.5 only) | 0.570 | 45.6% | +0.063 [+0.035, +0.091] | 0.521 / 59.7% |

Every target was seen before the rule was written, so this is not a confirmation. The planned CVC-300 holdout turned out to be a subset of CVC-ColonDB (task 12), so no independent test exists yet.

### 4.14 Phase 2 (2026-11-13): source-only continuation and DACS ablations

Scope fixed in the protocol (amendment_v2, phase-2 note) before any run.

**Source-only continued training** (`results/benchmark/controls.md`) uses the control's source batches without the target forward, so target data never enters the network.

- **Target images in BatchNorm, without a target loss, hurt.** Comparing the control with source-only continuation:
  - Dice: ColonDB −0.079 [−0.137, −0.027], FAR −0.047 [−0.077, −0.023].
  - Normal FPR: PolypGen +50.6 points, FAR +18.5 points.
  - The control's 74–79% FPR therefore comes from target BatchNorm statistics, not from training longer. Source-only continuation stays near SourceOnly: 26.9% on PolypGen and 54.3% on FAR.
- **More source training helps by itself.** Source-only continuation beats SourceOnly on ClinicDB (+0.045), ColonDB (+0.070) and FAR (+0.031).
- **Against this stronger, target-free reference,** methods whose interval excludes zero:
  - DACS on 4/5 targets (all but ETIS);
  - MinEnt+MMD on 3/5 (ETIS, ColonDB, PolypGen);
  - FDA on 2/5 (ETIS, PolypGen).
- **FPR against this reference:**
  - DACS is 9.0 points lower on FAR but 9.1 points higher on PolypGen.
  - MinEnt+MMD is higher on both.
  - FDA is level.
- **Caveat:** the reference saves a checkpoint every 200 steps, while the methods on PolypGen and FAR save every 400, which slightly favours the reference.

**DACS ablations** (`results/benchmark/ablations.md`) at λ=0.5 on ClinicDB and FAR:

| Model | ClinicDB Dice | ClinicDB small | FAR Dice | FAR small | FAR normal FPR |
|---|---:|---:|---:|---:|---:|
| DACS | 0.690 | 0.457 | 0.565 | 0.356 | 45.0% |
| without paste | 0.700 | 0.512 | 0.570 | 0.365 | 36.3% |
| without confidence weight | 0.690 | 0.458 | 0.570 | 0.363 | 44.7% |

- **The cross-domain paste does not help.**
  - Dice is unchanged without it.
  - Small-polyp Dice on ClinicDB is higher without it: DACS minus the variant is −0.075 [−0.149, −0.014].
  - FAR normal FPR is lower without it: DACS is +8.1 points [+2.7, +13.4] higher.
- **The confidence weight barely matters.**
- **What works in DACS is EMA-teacher hard pseudo-label self-training with colour jitter.**
- **The variant without the paste improves both Dice and FPR on FAR.** This is a post hoc finding on two targets at one λ.

## 5. In progress

Nothing running. Phases 0–2 finished 2026-11-13 (4.13, 4.14). Phase 3 as planned is invalid: CVC-300 is a subset of CVC-ColonDB (60/60 near-duplicates), so an independent holdout needs a dataset not yet on disk.

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
| 9 | ~~Amendment v2, CPU part: metric correction, H2 direct comparisons, endpoints, unified selection table, post hoc deployment rule~~ (done, see 4.13) | Credibility | CPU | Nov 13 |
| 10 | ~~Source-only continued training (6 runs, no target forward), scored on all targets~~ (done, see 4.14) | Separates more source training from target BatchNorm; FDA's proper control | ≈ 0.5 h GPU | Nov 13 |
| 11 | ~~DACS ablations (no paste; no confidence weight) on ClinicDB and FAR, λ=0.5~~ (done, see 4.14); MinEnt+MMD balance dropped by the protocol note | Mechanism | ≈ 1 h GPU | Nov 13 |
| 12 | Frozen-rule holdout. **CVC-300 cancelled:** a near-duplicate check matches all 60 images to CVC-ColonDB frames, which were already used for adaptation and evaluation. PolypGen positive sequences share videos with the evaluated single frames. A valid holdout needs a new dataset. | Independent confirmation | ≈ 0.5 h GPU once data exist | pending decision |
| 13 | Reproducibility: data manifests with hashes, split code back, deployed-model list, per-image CSVs | Reproducibility | CPU | phase 4 |
| 14 | Write-up: method, results, limitations; the earlier 18 attempts move to an appendix ([失败方法记录.md](失败方法记录.md)) | Report | – | phase 5 |

**Decision rules fixed now.**
- A method counts as helpful on a target only if its paired gain over the control is positive in at least 2/3 of trajectories, and its interval excludes zero or it does so on at least two targets.
- A method that raises the normal-image false-positive rate by more than 5 points on PolypGen or FAR is reported as a trade-off, not a success.

## 7. Limitations and risks

- **Few independent runs.** The real sample size is 6 base runs, since λ values within a run are correlated. Intervals are wide; MinEnt's interval against the control still includes zero.
- **Transductive targets.** ETIS, ColonDB, PolypGen and FAR use the same images for adaptation and test. That is standard in UDA, but those results are optimistic compared with ClinicDB's held-out test split.
- **Colonoscopy only.** "Domains" means hospitals, devices and cohorts within colonoscopy, not other imaging modalities.
- **Strict evaluation.** Small model, native resolution, and plain per-image Dice. Absolute numbers are lower than in papers that train on ClinicDB or evaluate at 352×352; conclusions rest on paired gains, not absolute Dice.
- **What the control controls for.** The control forwards source and target images together, and target images in BatchNorm turned out to hurt on its own (4.14), so gains over the control are inflated. Source-only continued training is the stronger, target-free reference. Every method with a joint forward pass carries the same BatchNorm mixing; that effect is not separated per method.
- **Development evidence vs confirmation.** ClinicDB and FAR were used for method development before 2026-11-01; PolypGen's λ 0.1/1 runs and the deployment rule were added after seeing results; four targets are transductive. None of this is an independent confirmation, and the planned CVC-300 holdout is a subset of CVC-ColonDB (task 12).
- **Validator coverage.** The 19-validator comparison holds only for ClinicDB's v1 pool; the current pools have 9 validators scored for every candidate.
- **Not done on purpose:** architecture comparisons (UNet++, nnU-Net), which would change the fixed network and the research question.
