# Qualitative examples

Rule-selected images (benchmark/qualitative.py docstring), chosen with the shown models: base run a0_seed42, Source-Risk pick, λ 0.5. Blue = prediction, white contour = ground truth; titles give that model's native Dice (FP/clean on normal images).

## ClinicDB

| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |
|---|---|---:|---|
| gain | 380.png | 1.65% | 0.29 / 0.81 |
| typical | 513.png | 7.04% | 0.88 / 0.90 |
| failure | 80.png | 1.17% | 0.54 / 0.24 |
| small | 191.png | 1.15% | 0.58 / 0.79 |

## ETIS

| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |
|---|---|---:|---|
| gain | 139.png | 1.38% | 0.11 / 0.42 |
| typical | 53.png | 0.75% | 0.04 / 0.00 |
| failure | 129.png | 0.73% | 0.90 / 0.87 |
| small | 73.png | 0.31% | 0.00 / 0.00 |

## ColonDB

| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |
|---|---|---:|---|
| gain | 181.png | 3.49% | 0.47 / 0.87 |
| typical | 231.png | 2.38% | 0.84 / 0.79 |
| failure | 115.png | 2.43% | 0.12 / 0.10 |
| small | 269.png | 1.73% | 0.53 / 0.42 |

## PolypGen

| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |
|---|---|---:|---|
| gain | C3_EndoCV2021_00190.jpg | 3.50% | 0.39 / 0.85 |
| typical | C3_EndoCV2021_00146.jpg | 11.04% | 0.01 / 0.37 |
| failure | 590OLCV1_100H0006.jpg | 6.08% | 0.53 / 0.12 |
| small | EndoCV2021_001212.jpg | 1.82% | 0.31 / 0.17 |
| normal ok | seq14_neg_00119.jpg | 0.00% | – |
| normal fp | seq14_neg_001.jpg | 0.00% | – |

## FAR

| Row | Image | GT fraction | Mean Dice SourceOnly / DACS (all picks) |
|---|---|---:|---|
| gain | 1403-10-3011231011-7477.jpg | 10.26% | 0.43 / 0.94 |
| typical | 1402-03-07103827-4739.jpg | 2.37% | 0.50 / 0.71 |
| failure | 1403-12-121322544-7647.jpg | 0.95% | 0.45 / 0.30 |
| small | 1401-08-151031124-3860.jpg | 1.50% | 0.31 / 0.24 |
| normal ok | 1402-07-29933113-5330.jpg | 0.00% | – |
| normal fp | 1402-09-051055473-5529.jpg | 0.00% | – |
