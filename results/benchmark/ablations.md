# Phase 2: DACS ablations (amendment v2)

λ = 0.5 only; checkpoint spacing as the existing DACS runs (ClinicDB, ETIS, ColonDB 200; PolypGen, FAR 400 steps). Without paste: teacher pseudo-label loss on jittered target images. Without confidence weight: weight 1 outside the pasted polyp. Paired differences DACS − variant, same base run; 95% intervals.

## ClinicDB

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 20 video groups.

| Model (λ 0.5) | Dice | Small-polyp Dice |
|---|---:|---:|
| DACS | 0.690 | 0.457 |
| DACS without paste | 0.700 | 0.512 |
| DACS without confidence weight | 0.690 | 0.458 |

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| DACS − DACS without paste | -0.006 [-0.028, +0.017] | -0.075 [-0.149, -0.014] |
| DACS − DACS without confidence weight | -0.001 [-0.006, +0.003] | -0.009 [-0.040, +0.012] |

## ETIS

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: single images: no video ids, so adjacent frames count as independent and intervals may be too narrow.

| Model (λ 0.5) | Dice | Small-polyp Dice |
|---|---:|---:|
| DACS | 0.414 | 0.271 |
| DACS without paste | 0.431 | 0.298 |

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| DACS − DACS without paste | -0.017 [-0.044, +0.008] | -0.027 [-0.069, +0.009] |

## ColonDB

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: single images: no video ids, so adjacent frames count as independent and intervals may be too narrow.

| Model (λ 0.5) | Dice | Small-polyp Dice |
|---|---:|---:|
| DACS | 0.624 | 0.386 |
| DACS without paste | 0.580 | 0.377 |

| Comparison | Dice | Small-polyp Dice |
|---|---|---|
| DACS − DACS without paste | +0.044 [+0.016, +0.072] | +0.010 [-0.024, +0.045] |

## PolypGen

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 6 centres for polyp images, 23 negative sequences for normal images.

| Model (λ 0.5) | Dice | Small-polyp Dice | Normal FPR |
|---|---:|---:|---:|
| DACS | 0.618 | 0.353 | 25.5% |
| DACS without paste | 0.624 | 0.366 | 16.1% |

| Comparison | Dice | Small-polyp Dice | Normal FPR (points) |
|---|---|---|---|
| DACS − DACS without paste | -0.008 [-0.018, +0.001] | -0.011 [-0.035, +0.012] | +9.0 [+0.7, +20.9] |

## FAR

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups here: 444 duplicate-merged patient groups.

| Model (λ 0.5) | Dice | Small-polyp Dice | Normal FPR |
|---|---:|---:|---:|
| DACS | 0.565 | 0.356 | 45.0% |
| DACS without paste | 0.570 | 0.365 | 36.3% |
| DACS without confidence weight | 0.570 | 0.363 | 44.7% |

| Comparison | Dice | Small-polyp Dice | Normal FPR (points) |
|---|---|---|---|
| DACS − DACS without paste | -0.006 [-0.021, +0.010] | -0.008 [-0.033, +0.018] | +8.1 [+2.7, +13.4] |
| DACS − DACS without confidence weight | -0.006 [-0.012, -0.000] | -0.008 [-0.019, -0.000] | +0.2 [-1.0, +1.3] |
