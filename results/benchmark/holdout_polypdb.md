# Phase 3: frozen-rule holdout on PolypDB Hanoi (amendment v2)

1000 WLI polyp images from Hanoi Medical University / IGH (Fujinon), never used before; transductive like the other targets; bootstrap over patient groups (Dice: 256 patients; small-polyp Dice: the 110 patients with small polyps). Rule and analysis fixed in a commit before any prediction on these images; evaluated once.

## Primary test

DACS λ=1, Source-Risk pick per trajectory, paired Dice gain over the same base run's SourceOnly pick: **+0.035 [+0.023, +0.049]** (6/6 runs better) → **confirmed** (criterion: interval above zero).

## Levels

| Model | Mean Dice (6 runs) | Small-polyp Dice | Single deployed model (run, step): Dice |
|---|---:|---:|---|
| SourceOnly | 0.751 | 0.528 | a1_seed44 step 23: 0.771 |
| Source-only continuation | 0.774 | 0.547 | a1_seed42 step 1600: 0.784 |
| DACS λ=1 | 0.788 | 0.575 | a1_seed42 step 1600: 0.795 |
| DACS without paste λ=0.5 | 0.790 | 0.570 | a1_seed42 step 400: 0.791 |

## Secondary comparisons (declared before running)

| Comparison | Dice [95% CI] | Small-polyp Dice [95% CI] |
|---|---|---|
| DACS λ=1 − Source-only continuation | +0.012 [+0.003, +0.022] | +0.040 [+0.009, +0.074] |
| DACS without paste λ=0.5 − SourceOnly | +0.037 [+0.023, +0.053] | +0.053 [+0.013, +0.098] |
| DACS without paste λ=0.5 − Source-only continuation | +0.014 [+0.006, +0.023] | +0.030 [+0.004, +0.059] |
| Source-only continuation − SourceOnly | +0.023 [+0.007, +0.040] | +0.023 [-0.017, +0.068] |
| Primary small-polyp gain (DACS λ=1 − SourceOnly) | – | +0.063 [+0.022, +0.108] |
