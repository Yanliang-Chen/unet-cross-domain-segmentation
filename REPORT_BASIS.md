# Report basis: unsupervised domain adaptation for polyp segmentation with a fixed U-Net

This file collects everything the written report needs: the question, the setup, every result with its source table, the limitations, the early work and the references. It is organised in report order. Numbers are copied from the tables in `results/benchmark/`; each section names its table, which is the authority if a number here ever disagrees.

Status: all experiments finished on 2026-11-13. Nothing is running. The report itself is not yet written.

---

## 0. Draft abstract

A U-Net trained on one hospital's colonoscopy images loses about 0.2 Dice on images from other hospitals and devices. We ask whether unsupervised domain adaptation (UDA), using only unlabelled target images, makes the *unchanged* U-Net (base width 16, depth 4, 256×256 input) segment polyps better than simply training it longer, and whether false positives on normal images rise as a result.

We train six source models on Kvasir-SEG and adapt each with eleven UDA methods, including two segmentation-specific ones (MinEnt+MMD and a DACS-style cross-domain mixing method). Five target domains are used: CVC-ClinicDB, ETIS-LaribPolypDB, CVC-ColonDB, PolypGen and FAR-POLYP-SEG. Every deployed model is picked by a rule fixed in advance (best source-validation Dice within the run) and compared, paired by source run, with SourceOnly and with a source-only continuation that never sees target data.

- **Dice:** DACS improves Dice over SourceOnly on all five targets, by +0.032 to +0.165, and over source-only continuation on four. Two tests frozen before any run confirm it:
  - an unused hospital (PolypDB Hanoi): +0.035 [+0.023, +0.049];
  - 222 FAR patients never used for adaptation: +0.042 [+0.013, +0.073] Dice with −8.2 [−15.5, −0.7] points of normal-image false positives against source-only continuation.
- **Mechanism:** ablations show that the gain comes from teacher pseudo-label self-training. The cross-domain paste helps only on one target and adds false positives.
- **Limits:**
  - false-positive behaviour is target-dependent: all methods raise it on PolypGen;
  - the matched "continued training" control is weak, because target images in BatchNorm hurt on their own;
  - choosing among methods without labels is unreliable, while choosing a checkpoint within one method's run is reliable.

*(Shorten to the report's word limit; keep the two frozen tests and the BatchNorm finding.)*

---

## 1. Introduction

### 1.1 Problem

Polyp segmentation models trained on one centre fail on other centres, devices and patient cohorts. Labelling every new site is expensive. UDA promises to use the new site's unlabelled images instead, but published gains are often measured against weak baselines, on the same images used for adaptation, and with model selection that quietly uses target labels.

### 1.2 Research question

> Can unsupervised domain adaptation, trained only with unlabelled target images, make the original U-Net segment polyps reliably better in new domains than simply training it longer — and can the improved model be chosen without target labels?

The network is fixed: the U-Net of Ronneberger et al., base width 16, depth 4, 256×256 input. Methods may change only the training loss or the pipeline. Discriminators and EMA teachers exist during training only, so the deployed model is always the plain U-Net. Architecture comparisons (UNet++, nnU-Net) are out of scope by design.

### 1.3 Hypotheses and final status

| | Hypothesis | Status | Where |
|---|---|---|---|
| H1 | Some adaptation methods beat a matched continued-training control. | Supported, but the control is weak (target images in BatchNorm hurt by themselves). Against the stronger source-only continuation, DACS still wins on 4/5 targets. | 4.1, 4.2 |
| H2 | Segmentation-specific methods (MinEnt+MMD, DACS) add to the best single methods. | Means support it; the pre-specified direct comparison does not (both beat every named comparator on 1/3 targets). The DACS paste helps only on ColonDB. | 4.3, 4.5 |
| H3 | Gains hold on other targets without more false positives on normal images. | Dice: supported, with two independent frozen tests. False positives: confirmed for DACS on FAR new patients only; on PolypGen (transductive) all three methods raise FPR over SourceOnly. | 4.1, 4.4, 4.6, 4.7 |
| H4 (supporting) | Label-free selection: source validation stays the safest rule. | Within a run, Source-Risk is reliable for the main methods (regret 0.004–0.032), failing mostly on ETIS and ColonDB. Across methods it is unreliable (ColonDB loss 0.224). | 4.10 |

### 1.4 Contributions (suggested wording)

1. A paired, multi-target evaluation of 11 UDA methods on a fixed U-Net with two references: a matched joint-forward control and a target-free source-only continuation. The evaluation measures Dice, small-polyp Dice and normal-image false positives.
2. The finding that target images passing through BatchNorm, without any target loss, hurt Dice and raise false positives by up to 51 points. "Continued training" controls therefore inflate UDA gains.
3. Two adapted methods for segmentation, with ablations showing that EMA-teacher pseudo-label self-training is the active part of DACS and that the cross-domain paste adds false positives.
4. Two confirmations frozen before any run: a new hospital and new patients. The new-patient test is the only one that checks false positives on data never used for adaptation.
5. An analysis of label-free checkpoint selection: where the deployment rule fails, and why selection across methods should not be used.

---

## 2. Related work (pointers)

- **Network.** U-Net [1].
- **UDA for segmentation:**
  - entropy-based: MinEnt and AdvEnt [6];
  - Fourier style transfer: FDA [7];
  - adversarial features: DANN [8];
  - kernel alignment: MMD/DAN [9];
  - normalisation statistics: AdaBN [10];
  - output-diversity losses: BNM [11], MCC [12];
  - teacher–student: Mean Teacher [13];
  - cross-domain mixing: DACS [5].
- **Label-free model selection:**
  - importance-weighted validation: IWCV [14], DEV [15];
  - neighbourhood density: SND [16];
  - mixed-sample probes: MixVal [17];
  - consistency ranking for biomedical segmentation: CMR [3].
- **Evaluating the complete UDA pipeline in medical imaging**, including selection and deployment, is argued for in [2]. Do not frame this project as a replication of that grid; it is a methods study that borrows the evaluation discipline.
- **Polyp datasets and reference U-Net numbers on unseen sets:** [4], [18]–[24].

---

## 3. Methods

### 3.1 Data

| Role | Dataset | Images | Setting |
|---|---|---|---|
| Source (labelled) | Kvasir-SEG [18] | 800 train / 200 validation | – |
| Target 1 | CVC-ClinicDB [19] | 185 unlabelled for adaptation / 427 labelled test, 20 video groups | inductive (adaptation and test images disjoint) |
| Target 2 | ETIS-LaribPolypDB [21] | 196 | transductive (same unlabelled images for adaptation and test) |
| Target 3 | CVC-ColonDB [20] | 380 | transductive |
| Target 4 | PolypGen C1–C6 [22] | 1,412 polyp + 125 empty-mask (polyp set, no polyp) + 1,755 normal (10 negative sequences) | transductive |
| Target 5 | FAR-POLYP-SEG [23] | 432 polyp + 7,749 normal, 444 duplicate-merged patients | transductive |
| Holdout | PolypDB, Hanoi centre, white light [24] | 1,000, 256 patients | transductive, frozen, evaluated once |
| New patients | FAR split by patient | adaptation: 222 patients, 4,016 images; test: 222 patients (198 polyp images from 93 patients, 3,967 normal) | inductive by patient, frozen, evaluated once |

- Target masks are never used for training or checkpoint selection.
- ETIS and ColonDB have no video ids.
- CVC-300 was planned as the holdout but is a subset of CVC-ColonDB (60/60 near-duplicates), so it was replaced by PolypDB Hanoi.
- PolypDB's Norwegian centre is excluded because it is the same hospital as Kvasir-SEG. No PolypDB Hanoi image has a near-copy in any data used before.

### 3.2 Source models

- Two augmentation settings × three seeds (42/43/44), 30 epochs each: 6 source runs, 180 checkpoints.
  - A0: flips.
  - A1: flips plus mild brightness, contrast and RGB gain.
- Final source-validation Dice is 0.78–0.83, in line with published U-Net results on Kvasir (about 0.82).
- On unseen ColonDB and ETIS, SourceOnly matches published U-Net numbers (about 0.51 and 0.40 [4]). So the drop is domain shift, not a weak baseline.

### 3.3 Adaptation training

- Every method starts from a source run's epoch-30 checkpoint and trains for 2,000 steps.
- Batches hold 8 source and 8 target images; the learning rate is 1e-4. The source BCE+Dice loss is kept.
- λ ∈ {0.1, 0.5, 1} weights the target loss.
- A checkpoint is saved every 200 steps (every 400 for the methods on PolypGen and FAR).
- A **trajectory** = method × source run × λ.
- FAR used λ=0.5 only. On PolypGen, λ=0.5 was pre-specified and λ 0.1/1 were added after seeing it; this is recorded in the protocol.

| Group | Methods |
|---|---|
| Matched control | **Continue**: same start, steps and joint source+target forward pass (target images enter BatchNorm), no target loss (λ=0) |
| Target-free reference (phase 2) | **Source-only continuation**: same source batches and steps, no target forward at all |
| Normalisation | AdaBN |
| Output level | MinEnt (pixel entropy), AdvEnt (adversarial entropy maps), pixel BNM, pixel MCC |
| Feature level | DANN (bottleneck gradient reversal), MMD (bottleneck multi-kernel) |
| Input level | FDA (low-frequency amplitude swap) |
| Teacher | Mean Teacher (EMA 0.99, teacher deployed) |
| Segmentation-specific (this project) | **MinEnt+MMD**: pixel entropy + bottleneck MMD. **DACS**: source polyp pixels pasted onto target images; source labels in the pasted region; EMA-teacher (0.99) hard pseudo-labels elsewhere, weighted by the share of confident teacher pixels (>0.968); colour jitter 0.2; student deployed |
| DACS ablations (phase 2) | **Without paste**: hard teacher pseudo-labels on jittered target images only. **Without confidence weight**: weight fixed at 1 |

Implementation: `benchmark/uda.py`.

### 3.4 Deployment rule (fixed in advance)

- Each trajectory deploys the checkpoint with the best source-validation Dice (**Source-Risk**).
- Each deployed model is paired with the same source run's control, its source-only continuation, and its SourceOnly pick (the source checkpoint chosen by the same rule).
- For a single deployable model, a post hoc rule picks the method and λ on ClinicDB alone; it selects **DACS λ=1** (4.9). That rule was then frozen and tested on the holdout and on new patients.

### 3.5 Metrics

All metrics are computed at native resolution: probabilities are upsampled bilinearly, predictions use > 0.5, and masks use > 127.

- **Primary:** mean per-image Dice on images with a non-empty mask.
- **Secondary:**
  - group-mean Dice and IoU;
  - small-polyp Dice (0 < GT area < 2% of the image);
  - normal-image false-positive rate (FPR): share of normal images with any predicted pixel, and with ≥ 0.1% of the image;
  - empty-mask FPR on PolypGen's 125 empty-mask images, reported separately;
  - image-level AUROC.
- **Metric correction (amendment v2).**
  - Version 1 counted PolypGen's 125 empty-mask images as positives, and as small polyps (area 0 < 2%).
  - Version 2 excludes them from Dice and reports them as their own FPR. Nothing changes on the other targets.
  - On PolypGen the correction means: DACS now beats SourceOnly, post-processing no longer appears to raise Dice, and small-polyp gains are +0.03 to +0.04 instead of near zero.
  - Old and new numbers: `results/benchmark/metric_correction.md`.

### 3.6 Statistics

- **Levels** (deployed Dice, small-polyp Dice, FPR) are per-image means over the test images, averaged over the 6 source runs.
- **Paired differences** compare a method with a reference trained from the same source run (and the same λ where both have one).
  - The difference is averaged within each group, then over the λ values of a run.
  - The 95% interval comes from 2,000 bootstrap draws over source runs and groups jointly (seed 20261101).
- Only groups that have images in an endpoint's image set enter that endpoint's bootstrap:

| Target | Dice | Small-polyp Dice | Normal FPR | Empty-mask FPR |
|---|---|---|---|---|
| ClinicDB | 20 videos | 9 videos | – | – |
| ETIS | 196 single images | 100 single images | – | – |
| ColonDB | 380 single images | 97 single images | – | – |
| PolypGen | 6 centres | 6 centres | 10 negative sequences | 6 centres |
| FAR | 186 patients with polyp images | 93 | 444 patients | – |
| PolypDB Hanoi | 256 patients | 110 | – | – |
| FAR new patients | 93 held-out patients with polyp images | 47 | 222 held-out patients | – |

- ETIS and ColonDB intervals treat adjacent frames as independent and may be too narrow.
- ClinicDB small-polyp Dice (9 videos) and PolypGen (6 centres) intervals are coarse.
- The real sample size is 6 source runs, because λ values within a run are correlated.

### 3.7 Decision rules and pre-registration

- **Helpful:** a method counts as helpful on a target if it beats the reference in at least 2/3 of trajectories, and its interval excludes zero or it does so on at least two targets.
- **Trade-off:** raising normal FPR by more than 5 points on PolypGen or FAR is reported as a trade-off, not a success.
- **Pre-registration:** every later rule was written into `benchmark_protocol.yaml` and committed before its numbers were computed:
  - amendment v2;
  - phase 2 scope;
  - holdout: commit 8646770;
  - no-paste extension: d0818ce;
  - new-patient test: ab537a8.

  Post hoc choices are labelled as such wherever they appear: PolypGen λ 0.1/1, the deployment rule, the no-paste variant, and the comparisons against source-only continuation at λ=0.5.

---

## 4. Results

### 4.1 Main results: deployed Dice per target

Mean of 6 source runs, Source-Risk pick per trajectory. Method rows average over λ 0.1/0.5/1 (FAR: λ=0.5 only); "without paste" is λ=0.5 only. Normal FPR is in brackets. Source: `results/benchmark/<target>/paired.md`.

| Method | ClinicDB | ETIS | ColonDB | PolypGen | FAR | PolypDB Hanoi (holdout) |
|---|---:|---:|---:|---:|---:|---:|
| DACS | 0.686 | 0.403 | 0.609 | 0.617 (37%) | 0.565 (45%) | 0.788 (λ=1) |
| DACS without paste (λ=0.5) | 0.700 | 0.431 | 0.580 | 0.624 (16%) | 0.570 (36%) | 0.790 |
| MinEnt+MMD | 0.670 | 0.412 | 0.568 | 0.615 (40%) | 0.542 (65%) | – |
| FDA | 0.650 | 0.430 | 0.508 | 0.629 (37%) | 0.563 (54%) | – |
| MinEnt | 0.664 | 0.393 | 0.534 | – | – | – |
| MMD | 0.658 | 0.388 | 0.541 | – | – | – |
| Source-only continuation | 0.654 | 0.379 | 0.514 | 0.601 (27%) | 0.533 (54%) | 0.774 |
| Continued-training control | 0.637 | 0.338 | 0.435 | 0.615 (79%) | 0.492 (74%) | – |
| SourceOnly | 0.610 | 0.366 | 0.444 | 0.582 (25%) | 0.510 (57%) | 0.751 |

**Paired gain over SourceOnly** [95% CI] (trajectories better / total):

| Target | DACS | MinEnt+MMD | FDA | Source-only continuation |
|---|---|---|---|---|
| ClinicDB | **+0.082** [+0.034, +0.138] (18/18) | **+0.068** [+0.020, +0.126] (16/18) | +0.042 [−0.007, +0.094] (15/18) | **+0.045** [+0.003, +0.093] |
| ETIS | **+0.038** [+0.014, +0.065] (16/18) | **+0.046** [+0.017, +0.080] (17/18) | **+0.064** [+0.033, +0.100] (18/18) | +0.014 [−0.013, +0.041] |
| ColonDB | **+0.165** [+0.135, +0.197] (18/18) | **+0.124** [+0.086, +0.168] (18/18) | **+0.064** [+0.029, +0.101] (17/18) | **+0.070** [+0.033, +0.108] |
| PolypGen | **+0.032** [+0.014, +0.052] (17/18) | **+0.029** [+0.014, +0.045] (18/18) | **+0.048** [+0.021, +0.074] (18/18) | +0.016 [−0.001, +0.033] |
| FAR | **+0.063** [+0.036, +0.092] (6/6) | **+0.038** [+0.011, +0.065] (6/6) | **+0.050** [+0.020, +0.084] (6/6) | **+0.031** [+0.003, +0.058] |

- DACS and MinEnt+MMD beat SourceOnly on 5/5 targets, FDA on 4/5 (not ClinicDB).
- The best method depends on the target:
  - DACS is first on ClinicDB, ColonDB and FAR;
  - FDA is first on ETIS and PolypGen;
  - MinEnt+MMD is in the top three everywhere.
- Gains are largest on ColonDB, the hardest shift relative to SourceOnly.
- The full 11-method grid exists for ClinicDB only (Table: `clinicdb/paired.md`):
  - beat the control with an interval above zero: DACS, MinEnt+MMD, MMD and DANN;
  - below the control: MCC (0.476), Mean Teacher, AdaBN and BNM;
  - AdvEnt: level with the control.
- Reduced grids on ETIS and ColonDB: every method beats the control there.

### 4.2 What the control controls for (H1)

Source: `results/benchmark/controls.md`; figure `figures/controls.png`.

**Gain over the continued-training control** (DACS / MinEnt+MMD / FDA):

| Target | Gain over control |
|---|---|
| ClinicDB | +0.050 / +0.036 / +0.011 |
| ETIS | +0.066 / +0.074 / +0.092 |
| ColonDB | +0.174 / +0.133 / +0.073 |
| PolypGen | +0.005 / +0.002 / +0.021 (all intervals include zero) |
| FAR | +0.079 / +0.054 / +0.066 |

**The control is weak.** Target images entering BatchNorm, with no target loss, hurt by themselves (control − source-only continuation):

| Target | Dice | Normal FPR (points) |
|---|---|---|
| ClinicDB | −0.014 [−0.061, +0.030] | – |
| ETIS | −0.042 [−0.090, +0.003] | – |
| ColonDB | **−0.079** [−0.137, −0.027] | – |
| PolypGen | +0.010 [−0.009, +0.028] | **+50.6** [+36.8, +65.8] |
| FAR | **−0.047** [−0.077, −0.023] | **+18.5** [+15.0, +22.3] |

- The control's 74–79% FPR comes from target BatchNorm statistics, not from training longer.
- More source training helps by itself: source-only continuation beats SourceOnly on ClinicDB (+0.045), ColonDB (+0.070) and FAR (+0.031).

**Against the stronger, target-free reference** (method − source-only continuation; Dice, and FPR in points):

| Target | DACS | MinEnt+MMD | FDA |
|---|---|---|---|
| ClinicDB | **+0.037** [+0.007, +0.066] | +0.023 [−0.006, +0.053] | −0.003 [−0.042, +0.039] |
| ETIS | +0.024 [−0.002, +0.052] | **+0.033** [+0.006, +0.064] | **+0.051** [+0.019, +0.084] |
| ColonDB | **+0.095** [+0.072, +0.122] | **+0.054** [+0.035, +0.076] | −0.006 [−0.027, +0.016] |
| PolypGen | **+0.016** [+0.004, +0.029]; FPR **+9.1** | **+0.012** [+0.004, +0.021]; FPR **+13.0** | **+0.031** [+0.013, +0.050]; FPR +8.5 |
| FAR | **+0.032** [+0.009, +0.055]; FPR **−9.0** | +0.007 [−0.014, +0.028]; FPR **+10.4** | +0.020 [−0.014, +0.053]; FPR +0.6 |

- Intervals above zero: DACS on 4/5 targets, MinEnt+MMD on 3/5, FDA on 2/5.
- Caveat: the reference saves a checkpoint every 200 steps, while the methods on PolypGen and FAR save every 400. This slightly favours the reference.
- Every method with a joint source+target forward pass carries the same BatchNorm mixing. That effect is not separated per method.

**Post hoc, on the targets already examined (not independent evidence).** Both DACS variants at λ=0.5 against source-only continuation:
- **DACS without paste:** above zero on 5/5 targets, Dice +0.023 to +0.066. FPR is lower on PolypGen (−10.7 [−15.3, −5.9]) and FAR (−17.0 [−20.3, −13.5]).
- **Full DACS λ=0.5:** above zero on 4/5 targets; PolypGen's lower bound is −0.000.

The independent test of this is 4.7.

### 4.3 H2: direct comparisons with single methods

Source: `results/benchmark/h2.md`. Pre-specified: supported if the interval excludes zero against every named comparator on at least 2 of 3 targets.

| Method | Comparator | ClinicDB | ETIS | ColonDB |
|---|---|---|---|---|
| MinEnt+MMD | MinEnt | +0.010 [−0.016, +0.034] | +0.019 [−0.003, +0.042] | **+0.034** [+0.020, +0.049] |
| MinEnt+MMD | MMD | +0.011 [−0.006, +0.033] | **+0.024** [+0.011, +0.040] | **+0.027** [+0.013, +0.042] |
| DACS | MinEnt | +0.024 [−0.005, +0.049] | +0.011 [−0.011, +0.035] | **+0.076** [+0.052, +0.099] |
| DACS | MMD | **+0.025** [+0.003, +0.048] | +0.016 [−0.004, +0.035] | **+0.068** [+0.048, +0.087] |
| DACS | FDA | **+0.040** [+0.004, +0.075] | **−0.027** [−0.053, −0.002] | **+0.101** [+0.077, +0.126] |

- Both methods beat every named comparator only on ColonDB (1/3 targets).
- H2 is "means support it, direct comparison inconclusive".
- DACS is significantly worse than FDA on ETIS.

### 4.4 Full endpoints and false positives (H3)

Sources: `results/benchmark/endpoints.md`; figure `figures/dice_fpr.png`.

**Endpoints against SourceOnly:**

| Target | DACS | MinEnt+MMD | FDA | Control |
|---|---|---|---|---|
| ClinicDB | Dice better | Dice better | not enough evidence | not enough evidence |
| ETIS | Dice better | Dice better | Dice better | not enough evidence |
| ColonDB | Dice better | Dice better | Dice better | not enough evidence |
| PolypGen | trade-off (Dice +0.032, FPR +11.2 pts) | trade-off (+0.029, +15.1) | trade-off (+0.048, +10.7) | trade-off (+0.027, +52.7) |
| FAR | **both** (Dice +0.063, FPR −10.8 pts) | trade-off (+0.038, +8.5) | Dice better (FPR −1.3 [−8.9, +5.9]) | FPR higher (+16.6) |

**Small-polyp Dice:**
- Intervals above zero:
  - all three methods on ColonDB and PolypGen;
  - DACS only on FAR;
  - FDA only on ETIS;
  - none on ClinicDB, which has 36 small polyps.
- None of the three methods significantly lowers small-polyp Dice.
- Gain over the same run's SourceOnly pick (trajectories won):

| Target | DACS | MinEnt+MMD | FDA |
|---|---|---|---|
| ETIS | +0.036 (14/18) | +0.037 (13/18) | +0.077 (18/18) |
| ColonDB | +0.116 (16/18) | +0.083 (16/18) | +0.087 (18/18) |
| FAR | +0.063 (6/6) | +0.031 (5/6) | +0.045 (6/6) |
| PolypGen (293 small polyps) | +0.041 (16/18) | +0.034 (14/18) | +0.036 (17/18) |

**Empty-mask FPR (PolypGen):** all intervals include zero.

**λ controls false positives on PolypGen.**
- DACS at λ 0.1 / 0.5 / 1: FPR 60.2% / 25.5% / 24.7%, Dice 0.610 / 0.618 / 0.625.
- So at λ ≥ 0.5 DACS matches SourceOnly's FPR with higher Dice.
- MinEnt+MMD behaves the same way; FDA does not (35–40%).

### 4.5 What works inside DACS (ablations)

Sources: `results/benchmark/ablations.md`; figure `figures/ablation.png`.

**Phase-2 ablations** at λ=0.5 on ClinicDB and FAR:

| Model | ClinicDB Dice | ClinicDB small | FAR Dice | FAR small | FAR normal FPR |
|---|---:|---:|---:|---:|---:|
| DACS | 0.690 | 0.457 | 0.565 | 0.356 | 45.0% |
| without paste | 0.700 | 0.512 | 0.570 | 0.365 | 36.3% |
| without confidence weight | 0.690 | 0.458 | 0.570 | 0.363 | 44.7% |

**No-paste variant extended to all five targets** (scope committed beforehand, d0818ce). DACS − without paste, λ=0.5:

| Target | Dice | Small-polyp Dice | Normal FPR (points) |
|---|---|---|---|
| ClinicDB | −0.006 [−0.028, +0.017] | **−0.075** [−0.149, −0.014] | – |
| ETIS | −0.017 [−0.044, +0.008] | −0.027 [−0.069, +0.009] | – |
| ColonDB | **+0.044** [+0.016, +0.072] | +0.010 [−0.024, +0.045] | – |
| PolypGen | −0.008 [−0.018, +0.001] | −0.011 [−0.035, +0.012] | **+9.0** [+0.7, +20.9] |
| FAR | −0.006 [−0.021, +0.010] | −0.008 [−0.033, +0.018] | **+8.1** [+2.7, +13.4] |

- The cross-domain paste raises Dice only on ColonDB.
- It raises normal FPR on both targets that have normal images, and lowers small-polyp Dice on ClinicDB.
- The confidence weight barely matters: FAR Dice difference −0.006 [−0.012, −0.000], FPR +0.2 points.
- **The active part of DACS is EMA-teacher hard pseudo-label self-training with colour jitter.** This differs from Mean Teacher with an MSE consistency target, which was a negative result on ClinicDB.
- Caveat: the no-paste variant was proposed after seeing ClinicDB and FAR, and was tested at one λ only.

### 4.6 Independent confirmation 1: new hospital (PolypDB Hanoi)

Source: `results/benchmark/holdout_polypdb.md`; figure `figures/holdout.png`. Frozen before any prediction (8646770), evaluated once. Transductive; 1,000 white-light images from 256 patients (Fujinon); patient-group bootstrap.

| Model | Mean Dice (6 runs) | Small-polyp Dice | Single deployed model |
|---|---:|---:|---:|
| SourceOnly | 0.751 | 0.528 | 0.771 |
| Source-only continuation | 0.774 | 0.547 | 0.784 |
| **DACS λ=1 (frozen rule)** | **0.788** | **0.575** | **0.795** |
| DACS without paste λ=0.5 | 0.790 | 0.570 | 0.791 |

- **Primary test passed:** DACS λ=1 − SourceOnly +0.035 [+0.023, +0.049], better in 6/6 runs; small polyps +0.063 [+0.022, +0.108].
- **Secondary (declared in advance):**
  - DACS λ=1 − source-only continuation: +0.012 [+0.003, +0.022], small polyps +0.040 [+0.009, +0.074];
  - DACS without paste: +0.037 over SourceOnly, +0.014 over source-only continuation.
- **Limits:**
  - one new centre;
  - transductive;
  - no normal images, so FPR is not tested;
  - SourceOnly is already high, so the gain is smaller than on ETIS or ColonDB.

### 4.7 Independent confirmation 2: new patients (FAR patient split)

Source: `results/benchmark/new_patients_far.md`; figure `figures/new_patients.png`. Frozen before any training (ab537a8), evaluated once.

**Why this test.** Every earlier FPR result is transductive: the model may only have learned the normal images it was later tested on. Here FAR's 444 patients are split 50/50, stratified within the 186 polyp patients and the 258 normal-only patients (seed 20261113). Adaptation sees only one half's unlabelled images; Dice and FPR are measured on the other half.

**Design:**
- **Reference:** source-only continuation.
- **Primary:** DACS λ=1 (the deployment rule) and DACS without paste λ=0.5.
- **Criteria:** lower 95% bound of ΔDice > 0, **and** upper 95% bound of the normal-FPR increase ≤ +5 points.

**Pre-set criteria:**

| Model (vs source-only continuation) | Δ Dice [95% CI] | Δ normal FPR, points [95% CI] | Result |
|---|---|---|---|
| DACS λ=1 | **+0.042** [+0.013, +0.073] | **−8.2** [−15.5, −0.7] | both met: supported |
| DACS without paste λ=0.5 | **+0.036** [+0.010, +0.062] | **−12.7** [−15.7, −9.9] | both met: supported |

**Levels on the test patients (image means):**

| Model | Dice | Small-polyp Dice | Normal FPR (any pixel) | Normal FPR (area ≥ 0.1%) |
|---|---:|---:|---:|---:|
| SourceOnly | 0.508 | 0.280 | 56.2% | 51.6% |
| Source-only continuation | 0.546 | 0.302 | 54.4% | 48.9% |
| DACS λ=1 | 0.581 | 0.338 | 46.0% | 40.0% |
| DACS λ=0.5 | 0.578 | 0.340 | 48.3% | 41.9% |
| DACS without paste λ=0.5 | 0.578 | 0.353 | **41.0%** | **34.9%** |

- **Over SourceOnly:**
  - DACS λ=1: Dice +0.089 [+0.052, +0.130], FPR −9.6 [−18.0, −1.2];
  - DACS without paste: Dice +0.083 [+0.049, +0.117], FPR −14.1 [−21.5, −6.9].
- **Source-only continuation over SourceOnly:** Dice +0.046, FPR −1.3 [−6.5, +4.3].
- **The paste on new patients:** Dice +0.004 [−0.017, +0.024], FPR **+7.0** [+1.5, +13.0]. This reproduces its transductive effect (+8.1).
- **Limits:**
  - same hospital and device as the adaptation patients, so this tests new patients, not a new centre;
  - FAR was used for method development before 2026-11-01, and the no-paste variant was proposed after seeing FAR;
  - Dice rests on 93 patients;
  - only DACS was tested.

### 4.8 Small-component post-processing

Source: `results/benchmark/postprocessing.md`.

**Rule fixed in advance.**
- Remove predicted 8-connected components below 0.1% of the image (primary) or 0.41% (sensitivity: the smallest polyp component in the source masks).
- "Helpful" means FPR falls by ≥ 5 points with Dice falling by ≤ 0.01 on PolypGen and FAR.

| Method | PolypGen FPR raw → 0.1% → 0.41% | FAR FPR raw → 0.1% → 0.41% |
|---|---|---|
| DACS | 36.8% → 32.2% → 27.8% | 45.0% → 38.0% → 31.4% |
| MinEnt+MMD | 39.7% → 35.1% → 29.9% | 65.2% → 58.7% → 51.0% |
| FDA | 37.1% → 32.2% → 26.6% | 54.4% → 48.2% → 41.3% |
| Control | 79.1% → 73.9% → 67.3% | 73.7% → 68.5% → 62.7% |
| SourceOnly | 24.7% → 21.4% → 17.7% | 56.5% → 51.3% → 45.0% |

- **Primary threshold:**
  - FAR: FPR falls 5.1–6.9 points, which passes;
  - PolypGen: FPR falls 4.4–4.8 points, just short of 5;
  - Dice changes by at most ±0.001 everywhere.
- **Sensitivity threshold:** −6.9 to −14.0 points with Dice −0.007 to +0.002. It is reported as sensitivity only, because switching to it would be post hoc.
- Post-processing does not change the ranking or replace adaptation. Most false positives are larger than 0.1% of the image.

### 4.9 Deployment of a single model (post hoc rule)

Source: `results/benchmark/deployment.md`.

**Rule:**
- method and λ chosen on ClinicDB only, among {DACS, MinEnt+MMD, FDA} × {0.1, 0.5, 1}: it picks **DACS λ=1**;
- checkpoint: the Source-Risk pick;
- single model: the highest-Source-Risk pick across the 6 runs, which is a1_seed42 on every target.

| Target | Single model Dice | Normal FPR | 6-run gain vs SourceOnly [95% CI] | SourceOnly single model Dice / FPR |
|---|---:|---:|---|---|
| ClinicDB | 0.687 | – | +0.093 [+0.045, +0.151] | 0.684 / – |
| ETIS | 0.468 | – | +0.056 [+0.025, +0.087] | 0.374 / – |
| ColonDB | 0.683 | – | +0.214 [+0.187, +0.245] | 0.500 / – |
| PolypGen | 0.631 | 16.0% | +0.040 [+0.022, +0.061] | 0.616 / 30.8% |
| FAR (λ=0.5 only) | 0.570 | 45.6% | +0.063 [+0.035, +0.091] | 0.521 / 59.7% |

Every target here was seen before the rule was written. Its confirmations are 4.6 and 4.7.

### 4.10 Choosing checkpoints without labels (H4, supporting)

Source: `results/benchmark/selection.md`; v1 analysis `results/benchmark/clinicdb/clinicdb_full.md`.

**The deployment rule inside each trajectory.**
- Regret = the best saved checkpoint by target Dice − the Source-Risk pick.
- A failure is a regret ≥ 0.05. This is a descriptive threshold set after the curves analysis, not a test.

| Target | Mean regret, DACS / MinEnt+MMD / FDA | Failures (3 methods) | Models that fail more |
|---|---|---:|---|
| ClinicDB | 0.012 / 0.014 / 0.020 | 1/54 | MCC 9/18; SourceOnly, AdaBN 3/6 each |
| ETIS | 0.022 / 0.020 / 0.016 | **7/54** | MinEnt, DANN 5/18 each; control, SourceOnly 2/6 each |
| ColonDB | 0.025 / 0.016 / 0.032 | **5/54** | control 3/6 (0.069), SourceOnly 3/6, AdaBN 3/6; DANN 5/18 |
| PolypGen | 0.009 / 0.005 / 0.004 | 0/54 | AdaBN 1/6 |
| FAR | 0.009 / 0.011 / 0.005 | 0/18 | control 1/6 |

- The rule is reliable for the main methods.
- Failures cluster on the largest shifts (ETIS, ColonDB) and on unstable models, whose source-validation Dice and target Dice move apart.
- PolypGen and FAR methods saved a checkpoint only every 400 steps, so their regret is smaller by construction.

**Selection across methods** (not used for deployment). Pools:
- **SourceOnly only:** the 180 source checkpoints;
- **original protocol pool:** the 13 original pools run on that target;
- **complete pool:** adds source-only continuation and the DACS ablations.

| Target | Pool | Candidates | Oracle | Source-Risk pick | Loss [95% CI] |
|---|---|---:|---|---|---|
| ClinicDB | original | 2,220 | 0.727 (DACS) | 0.676 (control) | 0.051 [−0.020, 0.099] |
| ClinicDB | complete | 2,400 | 0.732 (DACS without paste) | 0.676 (control) | 0.055 [0.001, 0.090] |
| ETIS | original / complete | 1,500 / 1,620 | 0.479 (MinEnt) | 0.410 (DACS) | 0.068 |
| ColonDB | original / complete | 1,500 / 1,620 | 0.711 (DACS) | 0.488 (control) | **0.224** |
| PolypGen | original / complete | 660 / 750 | 0.656 (FDA) | 0.632 (DACS) | 0.025 |
| FAR | original | 480 | 0.592 (FDA) | 0.539 (control) | 0.053 [0.002, 0.060] |
| FAR | complete | 600 | 0.592 (FDA) | 0.588 (DACS without paste) | 0.004 [−0.043, 0.020] |

- Cross-method selection often picks a control checkpoint. Adding the later models changes only FAR, because the pool then contains a model that is good on both source and target.
- **Validators:**
  - 19 are implemented, but all 19 were scored only on ClinicDB's SourceOnly pool and its v1 1,860-candidate pool; the other pools have 9 fully scored validators.
  - Among those 9, Source-Risk has the smallest loss on 5/5 targets in the complete pool.
  - Confidence-based validators (entropy, InfoMax, BNM, MCC) pick confident-but-wrong MCC models (0.41–0.45).
  - Two-class CMR picks near-empty early checkpoints without a collapse guard.
- **Label budget on ClinicDB:** labelling 5% of the test images gives a loss of 0.038, slightly below Source-Risk; smaller budgets do not beat it.
- **Hence the main results fix the method and select only within a trajectory.**

### 4.11 Training length, λ and qualitative behaviour

Sources: `results/benchmark/curves.md`, `qualitative.md`; figures `lambda.png`, `step_curves.png`, `step_fpr.png`, `qualitative_<target>.png`.

**λ.**
- DACS improves monotonically with λ on every target with three λ values. At λ 0.1 / 1:
  - ClinicDB 0.669 / 0.699;
  - ETIS 0.374 / 0.422;
  - ColonDB 0.545 / 0.658;
  - PolypGen 0.610 / 0.625.
- FDA is best at λ=0.1 on ClinicDB, ColonDB and PolypGen.
- MinEnt+MMD is flat in λ.

**Training length.**
- All three methods pass SourceOnly within 200–400 steps.
- The control drifts below its start on ETIS, ColonDB and FAR.
- MinEnt+MMD peaks early (mean pick at steps 480–760); DACS keeps improving on ColonDB.
- The control's normal FPR climbs to 83–85% over training, while DACS on FAR falls to about 41%.

**Qualitative.** Images were chosen by a fixed rule, not by eye.
- The typical gain is recovering a polyp SourceOnly missed entirely (Dice 0 → ≥ 0.9).
- The control fails with large false regions on mucosa, instruments and glare.
- Adapted models fail by losing low-contrast small polyps that SourceOnly found, or by spreading onto instruments.
- DACS false positives on normal images are small blobs on over-exposed or yellow-tinted frames.

---

## 5. Discussion points

1. **Training longer and target BatchNorm are confounds.** About half of each method's gain over SourceOnly on ClinicDB is explained by more source training. Target images in BatchNorm hurt independently. A UDA evaluation needs a target-free continuation as its reference; the joint-forward "λ=0" control is not enough.
2. **Simple self-training beats mixing.**
   - DACS's benefit comes from EMA-teacher hard pseudo-labels.
   - The paste, the defining part of DACS, helps only where the shift is largest (ColonDB) and costs false positives.
   - For a fixed small U-Net, "DACS without paste" is the most balanced variant seen. It is post hoc on the original targets but passed the frozen new-patient test.
3. **False positives depend on the target.**
   - FAR: DACS lowers FPR, including on new patients.
   - PolypGen: every method raises it over SourceOnly (by 11–15 points), and λ is the lever that controls it.
   - Reporting Dice alone would hide this.
4. **Development data vs confirmation.**
   - ClinicDB and FAR were used for method development before 2026-11-01. PolypGen's extra λ values, the deployment rule and the no-paste variant are post hoc.
   - The two frozen tests are the confirmatory evidence. Write them as such, and the rest as development evidence.
5. **Selection.** Choosing a checkpoint within a fixed method is safe. Choosing across methods without labels is not, and is where most published pipelines are weakest.

## 6. Limitations

- **Few independent runs:** 6 source runs; λ values within a run are correlated. Intervals are wide.
- **Transductive targets:** ETIS, ColonDB, PolypGen and FAR use the same images for adaptation and test. ClinicDB's held-out split and the FAR new-patient test point the same way.
- **Missing video ids:** ETIS and ColonDB intervals may be too narrow. ClinicDB small-polyp Dice rests on 9 videos, PolypGen on 6 centres.
- **Weak matched control:** gains over it are inflated; source-only continuation is the stronger reference, and its denser checkpoints slightly favour it.
- **Independent FPR evidence is narrow:** one hospital (FAR), DACS only. PolypDB Hanoi has no normal images.
- **Validator coverage:** the 19-validator comparison holds only for ClinicDB's v1 pool.
- **Colonoscopy only:** domains are hospitals, devices and cohorts, not other modalities.
- **Strict evaluation:** small network, native resolution, plain per-image Dice. Absolute numbers are below papers that train on ClinicDB or evaluate at 352×352; conclusions rest on paired gains.
- **Fixed network:** no architecture comparison by design.
- **Reproducibility:** manifests, hashes, deployed-model lists and per-image results are in git. A full rerun on a new machine has not been done. `python -m benchmark.manifest verify` reproduces a paired gain (+0.082) from tracked files only.

## 7. Conclusion (draft)

On a fixed U-Net, unsupervised domain adaptation gives real, confirmed Dice gains on new hospitals and new patients. This holds only when measured against a reference that also trains longer, and the gains vary by target.

- **What works:** teacher pseudo-label self-training (DACS, best without the paste). It improved Dice on every target and lowered false positives on FAR, including for patients never seen during adaptation.
- **What does not:** a matched control that lets target images into BatchNorm, which overstates every method's benefit; and choosing among methods without labels.

---

## Appendix A. Early work (before 2026-10-31)

Before the current line of work, 18 approaches were tried. None met the goal of improving segmentation and normal-image FPR together. Their code and raw outputs have been removed; their numbers survive in git history (the deleted `失败方法记录.md` and earlier protocol files). Summary for an appendix table:

| Approach | Result | Why it stopped |
|---|---|---|
| Stability-based checkpoint selection, entropy fallback | Dice vs source-validation selection: A0 +0.042, A1 −0.033 | Low perturbation does not mean a good mask |
| Source-quality / empty-mask guard rules | Avoided some A1 losses, lost some A0 gains | No consistent advantage |
| Foreground CMR and source+CMR combinations | A1 ColonDB/ETIS −0.007/−0.009 | Not consistently better |
| Whole-image multi-scale input | ClinicDB/ColonDB/ETIS −0.028/−0.029/−0.011 | Resolution and scale confounded |
| BCE + Tversky loss | −0.029/−0.039/−0.006 | Recall not consistently improved |
| Fixed-canvas downscaling | ETIS +0.030 (small polyps +0.077); ClinicDB/ColonDB −0.019/−0.009 | Target-specific |
| AdaBN single model | Unstable across domains | BN statistics alone do not fix drift |
| Source-anchored Mean Teacher | 23/24 seed comparisons positive, with negative seeds and small-polyp losses | Not robust |
| Source/AdaBN/Mean Teacher ensemble | FAR primary comparison A0/A1 −0.028/−0.029 | Formally failed |
| Cross-method label-free selection pilot | Validators chose degenerate candidates | Selection gap; became H4 |
| PolypGen high-confidence pseudo-labels | Extra gain over strong source continuation ≈ +0.001 | Explained by longer training |
| FDA (early version) | ClinicDB +0.004/+0.001 vs strong control, intervals include zero | Dropped at the time; re-implemented in the benchmark |
| Adding normal source images | FPR −13.3/−11.4 pts, Dice −0.012/−0.022 | Failed the Dice-retention bound; class/source confounding |
| Same-source positive+negative pairing | FPR −6.7/−5.3 pts, Dice +0.008/−0.008 | Interval missed the −0.01 bound |
| Absolute global-feature presence head (SAM) | Rejected ~61% of ClinicDB positives, Dice −0.36/−0.37 | FPR fell only through misses |
| Full UDA with the local presence head | Dice +0.016/+0.022 vs strong source, A1 FPR +3.3 pts | Not a joint improvement |
| Strong-source foreground intersection | Dice interval included zero; FPR +3.7/+0.6 pts vs source ensemble | 8-way joint criterion failed |
| Normal-gradient conflict projection | 3 of 12 criteria passed; FPR +0.7/+0.8 pts | No advantage over matched source |

Lessons that carried over:
- selection cannot rescue candidates without a real advantage;
- whole-image FPR is barely visible in the mean BCE (one confident pixel flags an image);
- adding negatives from a new source confounds class with domain;
- reused development data cannot be renamed as independent confirmation.

## Appendix B. Protocol timeline

| Date | Event | Commit |
|---|---|---|
| 2026-11-01 | Focus moved to UDA methods on a fixed U-Net; benchmark protocol fixed | – |
| 2026-11-01/02 | ClinicDB full grid; ETIS, ColonDB, PolypGen, FAR; post-processing; curves; qualitative | – |
| 2026-11-13 | Amendment v2 written before computing: metric correction, H2 direct test, endpoints, selection table, deployment rule | – |
| 2026-11-13 | Phase 2: source-only continuation, DACS ablations | – |
| 2026-11-13 | Holdout frozen (PolypDB Hanoi), evaluated once | 8646770 |
| 2026-11-13 | No-paste extension scope frozen | d0818ce |
| 2026-11-13 | FAR new-patient test frozen, run, evaluated once | ab537a8 (freeze), 26d9f23 (result) |
| 2026-11-13 | Per-endpoint group counts corrected; H4 rewritten with complete pool | 845caec |

## Appendix C. Figures and tables for the report

| Figure (`results/benchmark/figures/`) | Content | Report section |
|---|---|---|
| `gains.png` | Paired Dice gain per method and target over SourceOnly, the control and source-only continuation | 4.1, 4.2 |
| `controls.png` | The control taken apart: target BatchNorm vs more source training | 4.2 |
| `dice_fpr.png` | Deployed Dice vs normal FPR on PolypGen and FAR | 4.4 |
| `ablation.png` | DACS minus no-paste on all targets | 4.5 |
| `holdout.png` | PolypDB Hanoi frozen test | 4.6 |
| `new_patients.png` | FAR new-patient test with the pre-set criteria | 4.7 |
| `lambda.png`, `step_curves.png`, `step_fpr.png` | λ sensitivity and training-length curves | 4.11 |
| `qualitative_<target>.png` | Rule-chosen examples, including failures | 4.11 |

Tables: `results/benchmark/` (`<target>/paired.md`, `controls.md`, `h2.md`, `endpoints.md`, `ablations.md`, `holdout_polypdb.md`, `new_patients_far.md`, `postprocessing.md`, `deployment.md`, `selection.md`, `curves.md`, `metric_correction.md`, `qualitative.md`). Per-image results: `results/per_image/`. Manifests and environment: `results/manifests/`. Pre-registered rules: `benchmark_protocol.yaml`.

## References

**Network and evaluation**
1. Ronneberger O., Fischer P., Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI, 2015. https://arxiv.org/abs/1505.04597
2. Xiong Y., Gallée L., Wolf D. S., Hillenhagen H., Götz M. How Far from Clinical Deployment? Evaluating the Complete Unsupervised Domain Adaptation Pipeline in Medical Imaging. arXiv:2608.12035, 2026.
3. Talks J., Marchesini K., Lumetti L., Bolelli F., Kreshuk A. Unsupervised Source-Free Ranking of Biomedical Segmentation Models Under Distribution Shift. arXiv:2503.00450, 2025.
4. Fan D.-P. et al. PraNet: Parallel Reverse Attention Network for Polyp Segmentation. MICCAI, 2020.

**Adaptation methods**

5. Tranheden W., Olsson V., Pinto J., Svensson L. DACS: Domain Adaptation via Cross-domain Mixed Sampling. WACV, 2021.
6. Vu T.-H., Jain H., Bucher M., Cord M., Pérez P. ADVENT: Adversarial Entropy Minimization for Domain Adaptation in Semantic Segmentation. CVPR, 2019.
7. Yang Y., Soatto S. FDA: Fourier Domain Adaptation for Semantic Segmentation. CVPR, 2020.
8. Ganin Y. et al. Domain-Adversarial Training of Neural Networks. JMLR, 2016.
9. Long M., Cao Y., Wang J., Jordan M. Learning Transferable Features with Deep Adaptation Networks. ICML, 2015.
10. Li Y., Wang N., Shi J., Hou X., Liu J. Adaptive Batch Normalization for Practical Domain Adaptation. Pattern Recognition, 2018.
11. Cui S. et al. Towards Discriminability and Diversity: Batch Nuclear-norm Maximization under Label Insufficient Situations. CVPR, 2020.
12. Jin Y., Wang X., Long M., Wang J. Minimum Class Confusion for Versatile Domain Adaptation. ECCV, 2020.
13. Tarvainen A., Valpola H. Mean teachers are better role models. NeurIPS, 2017.

**Label-free validators**

14. Sugiyama M., Krauledat M., Müller K.-R. Covariate Shift Adaptation by Importance Weighted Cross Validation. JMLR, 2007.
15. You K., Wang X., Long M., Jordan M. Towards Accurate Model Selection in Deep Unsupervised Domain Adaptation. ICML, 2019.
16. Saito K. et al. Tune it the Right Way: Unsupervised Validation of Domain Adaptation via Soft Neighborhood Density. ICCV, 2021.
17. Hu D. et al. Mixed Samples as Probes for Unsupervised Model Selection in Domain Adaptation. NeurIPS, 2023.

**Data**

18. Jha D. et al. Kvasir-SEG: A Segmented Polyp Dataset. MMM, 2020.
19. Bernal J. et al. WM-DOVA maps for accurate polyp highlighting in colonoscopy. Computerized Medical Imaging and Graphics, 2015. (CVC-ClinicDB)
20. Bernal J., Sánchez J., Vilariño F. Towards automatic polyp detection with a polyp appearance model. Pattern Recognition, 2012. (CVC-ColonDB)
21. Silva J. et al. Toward embedded detection of polyps in WCE images for early diagnosis of colorectal cancer. IJCARS, 2014. (ETIS-LaribPolypDB)
22. Ali S. et al. A multi-centre polyp detection and segmentation dataset for generalisability assessment. Scientific Data, 2023. (PolypGen)
23. FAR-POLYP-SEG: A Prospective Single-Center Colonoscopy Dataset for Colorectal Polyp Segmentation with Patient-Level Metadata and Baseline Cross-Dataset Evaluation. 2026. https://doi.org/10.1007/s10278-026-02268-5
24. Jha D. et al. PolypDB: A Curated Multi-Center Dataset for Development of AI Algorithms in Colonoscopy. 2024. https://osf.io/pr7ms/
