# Lambda, training length and small polyps

From results/benchmark/<target>/candidates.csv (benchmark/curves.py). Figures: `figures/step_curves.png`, `figures/step_fpr.png`, `figures/lambda.png`.

## 1. Lambda sensitivity (deployed Source-Risk pick, mean over 6 base runs)

Cells: Dice at λ 0.1 / 0.5 / 1; PolypGen also normal-image FPR. FAR has λ=0.5 only.

| Target | Method | Dice λ 0.1 | λ 0.5 | λ 1 | Normal FPR λ 0.1 | λ 0.5 | λ 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ClinicDB | DACS | 0.669 | 0.690 | 0.699 | – | – | – |
| ClinicDB | MinEnt+MMD | 0.666 | 0.674 | 0.671 | – | – | – |
| ClinicDB | FDA | 0.663 | 0.637 | 0.650 | – | – | – |
| ETIS | DACS | 0.374 | 0.414 | 0.422 | – | – | – |
| ETIS | MinEnt+MMD | 0.402 | 0.410 | 0.424 | – | – | – |
| ETIS | FDA | 0.431 | 0.434 | 0.425 | – | – | – |
| ColonDB | DACS | 0.545 | 0.624 | 0.658 | – | – | – |
| ColonDB | MinEnt+MMD | 0.550 | 0.579 | 0.575 | – | – | – |
| ColonDB | FDA | 0.538 | 0.502 | 0.485 | – | – | – |
| PolypGen | DACS | 0.590 | 0.601 | 0.606 | 60.2% | 25.5% | 24.7% |
| PolypGen | MinEnt+MMD | 0.604 | 0.603 | 0.601 | 52.1% | 34.0% | 32.9% |
| PolypGen | FDA | 0.622 | 0.611 | 0.601 | 35.3% | 36.4% | 39.7% |

## 2. Training length (all λ; FAR λ=0.5)

Per trajectory: start = warm start (SourceOnly epoch 30); last = step 2000; pick = Source-Risk pick; best = best saved checkpoint by target Dice (needs labels); regret = best − pick. Means over trajectories.

| Target | Method | Start | Last | Pick | Best | Regret | Mean pick step | Pick = last |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ClinicDB | DACS | 0.617 | 0.682 | 0.686 | 0.698 | 0.012 | 1211 | 2/18 |
| ClinicDB | MinEnt+MMD | 0.617 | 0.621 | 0.670 | 0.684 | 0.014 | 478 | 0/18 |
| ClinicDB | FDA | 0.617 | 0.642 | 0.650 | 0.670 | 0.020 | 1533 | 0/18 |
| ClinicDB | Control | 0.617 | 0.609 | 0.637 | 0.651 | 0.013 | 967 | 0/6 |
| ETIS | DACS | 0.351 | 0.397 | 0.403 | 0.426 | 0.022 | 1289 | 4/18 |
| ETIS | MinEnt+MMD | 0.351 | 0.380 | 0.412 | 0.432 | 0.020 | 722 | 1/18 |
| ETIS | FDA | 0.351 | 0.408 | 0.430 | 0.446 | 0.016 | 1400 | 0/18 |
| ETIS | Control | 0.351 | 0.308 | 0.338 | 0.379 | 0.041 | 1533 | 1/6 |
| ColonDB | DACS | 0.443 | 0.626 | 0.609 | 0.635 | 0.025 | 1333 | 5/18 |
| ColonDB | MinEnt+MMD | 0.443 | 0.496 | 0.568 | 0.584 | 0.016 | 567 | 0/18 |
| ColonDB | FDA | 0.443 | 0.512 | 0.508 | 0.540 | 0.032 | 1544 | 3/18 |
| ColonDB | Control | 0.443 | 0.410 | 0.435 | 0.504 | 0.069 | 1400 | 0/6 |
| PolypGen | DACS | 0.567 | 0.596 | 0.599 | 0.606 | 0.007 | 1378 | 1/18 |
| PolypGen | MinEnt+MMD | 0.567 | 0.561 | 0.603 | 0.605 | 0.003 | 756 | 0/18 |
| PolypGen | FDA | 0.567 | 0.601 | 0.611 | 0.615 | 0.003 | 1400 | 3/18 |
| PolypGen | Control | 0.567 | 0.574 | 0.592 | 0.600 | 0.008 | 1333 | 0/6 |
| FAR | DACS | 0.492 | 0.563 | 0.565 | 0.574 | 0.009 | 1600 | 1/6 |
| FAR | MinEnt+MMD | 0.492 | 0.505 | 0.542 | 0.553 | 0.011 | 733 | 0/6 |
| FAR | FDA | 0.492 | 0.551 | 0.563 | 0.568 | 0.005 | 1400 | 1/6 |
| FAR | Control | 0.492 | 0.447 | 0.492 | 0.513 | 0.021 | 1267 | 1/6 |

## 3. Small-polyp Dice (GT below 2% of the image; deployed picks, all λ)

Cells: small-polyp Dice (all-polyp Dice). Gain columns: mean over trajectories of the difference to the same base run's SourceOnly pick, with wins.

| Method | ClinicDB (n=36) | ETIS (n=100) | ColonDB (n=97) | PolypGen (n=418) | FAR (n=123) |
|---|---:|---:|---:|---:|---:|
| SourceOnly | 0.449 (0.610) | 0.222 (0.366) | 0.258 (0.444) | 0.358 (0.573) | 0.292 (0.510) |
| Control | 0.393 (0.637) | 0.159 (0.338) | 0.239 (0.435) | 0.341 (0.592) | 0.256 (0.492) |
| AdaBN | 0.270 (0.606) | 0.171 (0.334) | 0.227 (0.458) | 0.160 (0.469) | 0.190 (0.426) |
| DACS | 0.458 (0.686) | 0.258 (0.403) | 0.373 (0.609) | 0.364 (0.599) | 0.356 (0.565) |
| MinEnt+MMD | 0.453 (0.670) | 0.259 (0.412) | 0.341 (0.568) | 0.381 (0.603) | 0.323 (0.542) |
| FDA | 0.470 (0.650) | 0.299 (0.430) | 0.345 (0.508) | 0.366 (0.611) | 0.337 (0.563) |
| MinEnt | 0.478 (0.664) | 0.237 (0.393) | 0.345 (0.534) | – | – |
| MMD | 0.412 (0.658) | 0.220 (0.388) | 0.299 (0.541) | – | – |
| DANN | 0.416 (0.649) | 0.187 (0.362) | 0.294 (0.498) | – | – |

| Method | ClinicDB | ETIS | ColonDB | PolypGen | FAR |
|---|---|---|---|---|---|
| DACS gain vs SourceOnly | +0.009 (9/18; all +0.076) | +0.036 (14/18; all +0.038) | +0.116 (16/18; all +0.165) | +0.006 (10/18; all +0.026) | +0.063 (6/6; all +0.055) |
| MinEnt+MMD gain vs SourceOnly | +0.005 (7/18; all +0.060) | +0.037 (13/18; all +0.046) | +0.083 (16/18; all +0.124) | +0.023 (13/18; all +0.030) | +0.031 (5/6; all +0.033) |
| FDA gain vs SourceOnly | +0.021 (11/18; all +0.040) | +0.077 (18/18; all +0.064) | +0.087 (18/18; all +0.064) | +0.008 (10/18; all +0.038) | +0.045 (6/6; all +0.053) |
| Control gain vs SourceOnly | -0.055 (2/6; all +0.027) | -0.063 (1/6; all -0.028) | -0.019 (3/6; all -0.009) | -0.017 (2/6; all +0.019) | -0.036 (2/6; all -0.017) |
