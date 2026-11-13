# New-patient test on FAR (amendment v2)

Test patients never used for adaptation: 198 polyp images and 3967 normal images; adaptation used the unlabelled images of the other patients. **Levels** are image means over the test images (mean of 6 base runs). **Differences** are paired by base run and averaged per patient group, with 95% intervals from a joint bootstrap over base runs and patient groups. Rule frozen in a commit before any run; evaluated once.

## Pre-set criteria (reference: source-only continuation)

| Model | Δ Dice [95% CI] | Dice criterion (lower bound > 0) | Δ normal FPR, points [95% CI] | FPR criterion (upper bound ≤ +5) | H3 on new patients |
|---|---|---|---|---|---|
| DACS λ=1 (deployment rule) | +0.042 [+0.013, +0.073] | met | -8.2 [-15.5, -0.7] | met | **supported** |
| DACS without paste λ=0.5 | +0.036 [+0.010, +0.062] | met | -12.7 [-15.7, -9.9] | met | **supported** |

## Levels on the test patients (image means)

| Model | Dice | Small-polyp Dice | Normal FPR (any pixel) | Normal FPR (area ≥ 0.1%) |
|---|---:|---:|---:|---:|
| SourceOnly | 0.508 | 0.280 | 56.2% | 51.6% |
| Source-only continuation | 0.546 | 0.302 | 54.4% | 48.9% |
| DACS λ=1 | 0.581 | 0.338 | 46.0% | 40.0% |
| DACS λ=0.5 | 0.578 | 0.340 | 48.3% | 41.9% |
| DACS without paste λ=0.5 | 0.578 | 0.353 | 41.0% | 34.9% |

## Secondary paired differences (declared before running; patient groups)

| Comparison | Δ Dice | Δ small-polyp Dice | Δ normal FPR (points) | Δ area FPR (points) |
|---|---|---|---|---|
| DACS λ=1 − SourceOnly | +0.089 [+0.052, +0.130] | +0.065 [+0.013, +0.128] | -9.6 [-18.0, -1.2] | -10.6 [-19.0, -2.7] |
| DACS without paste λ=0.5 − SourceOnly | +0.083 [+0.049, +0.117] | +0.083 [+0.022, +0.154] | -14.1 [-21.5, -6.9] | -15.3 [-22.1, -8.7] |
| Source-only continuation − SourceOnly | +0.046 [+0.015, +0.080] | +0.037 [-0.022, +0.104] | -1.3 [-6.5, +4.3] | -2.2 [-7.0, +3.3] |
| DACS λ=0.5 − source-only continuation | +0.040 [+0.016, +0.067] | +0.035 [-0.002, +0.076] | -5.7 [-10.0, -1.0] | -6.6 [-11.4, -1.7] |
| DACS λ=0.5 − DACS without paste λ=0.5 (effect of the paste) | +0.004 [-0.017, +0.024] | -0.012 [-0.052, +0.021] | +7.0 [+1.5, +13.0] | +6.6 [+1.7, +12.0] |
