# Phase 2: source-only continued training (amendment v2)

Source-only continuation = the control's source batches without the target forward, so target images never reach the network (one set of 6 runs, scored on every target). Paired differences (same base run; lambdas averaged) with 95% intervals; normal FPR in percentage points.

## ClinicDB (source-only continuation: Dice 0.654, small 0.433)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 20 video groups.

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| Control − source-only continuation (target images in BatchNorm) | -0.014 [-0.061, +0.030] | -0.040 [-0.142, +0.048] |
| Source-only continuation − SourceOnly (more source training) | +0.045 [+0.002, +0.095] | -0.037 [-0.135, +0.055] |
| DACS − source-only continuation | +0.037 [+0.007, +0.066] | +0.048 [-0.069, +0.141] |
| MinEnt+MMD − source-only continuation | +0.023 [-0.006, +0.053] | +0.036 [-0.041, +0.110] |
| FDA − source-only continuation | -0.003 [-0.042, +0.039] | +0.059 [-0.073, +0.168] |
| DACS λ=0.5 − source-only continuation (post hoc) | +0.043 [+0.012, +0.076] | +0.050 [-0.066, +0.144] |
| DACS without paste λ=0.5 − source-only continuation (post hoc) | +0.049 [+0.023, +0.079] | +0.125 [+0.035, +0.227] |

## ETIS (source-only continuation: Dice 0.379, small 0.244)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: single images: no video ids, so adjacent frames count as independent and intervals may be too narrow.

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| Control − source-only continuation (target images in BatchNorm) | -0.042 [-0.090, +0.003] | -0.085 [-0.146, -0.026] |
| Source-only continuation − SourceOnly (more source training) | +0.014 [-0.013, +0.040] | +0.022 [-0.013, +0.056] |
| DACS − source-only continuation | +0.024 [-0.002, +0.052] | +0.013 [-0.023, +0.049] |
| MinEnt+MMD − source-only continuation | +0.033 [+0.006, +0.064] | +0.014 [-0.022, +0.064] |
| FDA − source-only continuation | +0.051 [+0.019, +0.084] | +0.055 [+0.009, +0.105] |
| DACS λ=0.5 − source-only continuation (post hoc) | +0.035 [+0.005, +0.066] | +0.026 [-0.016, +0.065] |
| DACS without paste λ=0.5 − source-only continuation (post hoc) | +0.052 [+0.028, +0.080] | +0.053 [+0.022, +0.090] |

## ColonDB (source-only continuation: Dice 0.514, small 0.335)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: single images: no video ids, so adjacent frames count as independent and intervals may be too narrow.

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| Control − source-only continuation (target images in BatchNorm) | -0.079 [-0.137, -0.027] | -0.097 [-0.175, -0.034] |
| Source-only continuation − SourceOnly (more source training) | +0.070 [+0.032, +0.107] | +0.078 [+0.023, +0.145] |
| DACS − source-only continuation | +0.095 [+0.072, +0.122] | +0.038 [+0.000, +0.075] |
| MinEnt+MMD − source-only continuation | +0.054 [+0.035, +0.076] | +0.006 [-0.024, +0.035] |
| FDA − source-only continuation | -0.006 [-0.027, +0.016] | +0.010 [-0.022, +0.043] |
| DACS λ=0.5 − source-only continuation (post hoc) | +0.110 [+0.083, +0.142] | +0.051 [+0.010, +0.096] |
| DACS without paste λ=0.5 − source-only continuation (post hoc) | +0.066 [+0.037, +0.095] | +0.041 [+0.006, +0.082] |

## PolypGen (source-only continuation: Dice 0.601, small 0.338, normal FPR 26.9%)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 6 centres for polyp images, 23 negative sequences for normal images.

| Comparison | Dice | Small-polyp Dice | Normal FPR (points) |
|---|---|---|---|
| Control − source-only continuation (target images in BatchNorm) | +0.010 [-0.009, +0.028] | +0.003 [-0.029, +0.033] | +50.6 [+36.8, +65.8] |
| Source-only continuation − SourceOnly (more source training) | +0.016 [-0.000, +0.032] | +0.025 [-0.010, +0.057] | +2.1 [-4.2, +8.4] |
| DACS − source-only continuation | +0.016 [+0.004, +0.029] | +0.016 [-0.001, +0.033] | +9.1 [+0.5, +20.7] |
| MinEnt+MMD − source-only continuation | +0.012 [+0.004, +0.021] | +0.007 [-0.012, +0.028] | +13.0 [+2.0, +25.7] |
| FDA − source-only continuation | +0.031 [+0.013, +0.050] | +0.019 [-0.011, +0.050] | +8.5 [-1.9, +18.0] |
| DACS λ=0.5 − source-only continuation (post hoc) | +0.016 [-0.000, +0.031] | +0.018 [-0.004, +0.040] | -1.7 [-11.2, +10.5] |
| DACS without paste λ=0.5 − source-only continuation (post hoc) | +0.023 [+0.013, +0.035] | +0.030 [+0.015, +0.049] | -10.7 [-15.3, -5.9] |

## FAR (source-only continuation: Dice 0.533, small 0.316, normal FPR 54.3%)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 444 duplicate-merged patient groups.

| Comparison | Dice | Small-polyp Dice | Normal FPR (points) |
|---|---|---|---|
| Control − source-only continuation (target images in BatchNorm) | -0.047 [-0.077, -0.023] | -0.062 [-0.095, -0.030] | +18.5 [+15.0, +22.3] |
| Source-only continuation − SourceOnly (more source training) | +0.031 [+0.003, +0.060] | +0.031 [-0.023, +0.091] | -1.9 [-6.5, +3.4] |
| DACS − source-only continuation | +0.032 [+0.009, +0.055] | +0.035 [-0.002, +0.077] | -9.0 [-13.8, -5.0] |
| MinEnt+MMD − source-only continuation | +0.007 [-0.014, +0.028] | -0.000 [-0.045, +0.042] | +10.4 [+2.4, +17.2] |
| FDA − source-only continuation | +0.020 [-0.014, +0.053] | +0.012 [-0.052, +0.074] | +0.6 [-5.8, +6.5] |
| DACS λ=0.5 − source-only continuation (post hoc) | +0.032 [+0.009, +0.055] | +0.035 [-0.002, +0.077] | -9.0 [-13.8, -5.0] |
| DACS without paste λ=0.5 − source-only continuation (post hoc) | +0.038 [+0.015, +0.063] | +0.043 [+0.010, +0.081] | -17.0 [-20.3, -13.5] |
