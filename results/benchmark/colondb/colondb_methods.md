# Selection gap: `colondb_methods` (pools: source_only, uda_adabn, uda_continue, uda_minent, uda_mmd, uda_dann, uda_fda, uda_minent_mmd, uda_dacs)

## clinicdb (n=1500)

Oracle Dice 0.727 (dacs_clinicdb_a1_seed44_l1_s0600); pool mean 0.621

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.676 | 0.051 [-0.020, 0.099] | 0.051 | 0.62 | 0.46 | – | 0.019 | 0.42 |
| entropy | minent | 0.675 | 0.052 [0.009, 0.091] | 0.052 | 0.66 | 0.49 | – | 0.034 | 0.21 |
| infomax | continue | 0.520 | 0.207 [0.114, 0.264] | 0.207 | 0.05 | 0.02 | – | 0.044 | -0.03 |
| bnm | minent | 0.527 | 0.200 [0.091, 0.277] | 0.200 | 0.10 | 0.06 | – | 0.043 | 0.01 |
| mcc | minent | 0.675 | 0.052 [0.009, 0.090] | 0.052 | 0.45 | 0.32 | – | 0.042 | -0.08 |
| cmr_nhd_gauss | source_only | 0.146 | 0.581 [0.448, 0.639] | 0.084 | 0.40 | 0.27 | – | 0.037 | 0.19 |
| cmr_ei_gauss | minent | 0.643 | 0.084 [0.028, 0.144] | 0.084 | 0.47 | 0.33 | – | 0.029 | 0.23 |
| cmr_nhd_fg_gauss | dacs | 0.670 | 0.057 [-0.038, 0.087] | 0.057 | 0.39 | 0.28 | – | 0.033 | 0.17 |
| cmr_ei_fg_gauss | minent | 0.660 | 0.067 [-0.025, 0.111] | 0.067 | 0.41 | 0.29 | – | 0.030 | 0.17 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.061; 2% (9 img) gap 0.050; 3% (13 img) gap 0.053; 4% (17 img) gap 0.047; 5% (21 img) gap 0.038

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.660 | 0.602 | 0.660 | 0.559 |
| continue | 60 | 0.696 | 0.676 | 0.676 | 0.655 |
| dacs | 180 | 0.727 | 0.698 | 0.726 | 0.670 |
| dann | 180 | 0.695 | 0.669 | 0.665 | 0.643 |
| fda | 180 | 0.697 | 0.668 | 0.648 | 0.677 |
| minent | 180 | 0.714 | 0.648 | 0.675 | 0.643 |
| minent_mmd | 180 | 0.713 | 0.663 | 0.610 | 0.642 |
| mmd | 180 | 0.707 | 0.635 | 0.645 | 0.639 |
| source_only | 180 | 0.684 | 0.684 | 0.622 | 0.575 |

Last-checkpoint reference: within-run gap 0.041.

## colondb (n=1500)

Oracle Dice 0.711 (dacs_colondb_a1_seed44_l1_s1800); pool mean 0.485

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.488 | 0.224 [0.195, 0.253] | 0.224 | 0.52 | 0.38 | – | 0.029 | 0.24 |
| entropy | minent_mmd | 0.194 | 0.517 [0.478, 0.555] | 0.145 | 0.61 | 0.43 | – | 0.045 | 0.15 |
| infomax | continue | 0.297 | 0.415 [0.382, 0.447] | 0.415 | -0.08 | -0.05 | – | 0.056 | -0.01 |
| bnm | continue | 0.297 | 0.415 [0.382, 0.447] | 0.415 | -0.01 | 0.01 | – | 0.054 | 0.03 |
| mcc | minent | 0.386 | 0.325 [0.294, 0.356] | 0.325 | 0.18 | 0.12 | – | 0.056 | -0.08 |
| cmr_nhd_gauss | source_only | 0.043 | 0.668 [0.635, 0.701] | 0.299 | 0.18 | 0.12 | – | 0.063 | 0.03 |
| cmr_ei_gauss | minent_mmd | 0.194 | 0.517 [0.482, 0.553] | 0.299 | 0.31 | 0.22 | – | 0.055 | 0.07 |
| cmr_nhd_fg_gauss | adabn | 0.291 | 0.421 [0.391, 0.450] | 0.421 | -0.04 | -0.03 | – | 0.055 | 0.02 |
| cmr_ei_fg_gauss | adabn | 0.291 | 0.421 [0.390, 0.451] | 0.421 | -0.03 | -0.03 | – | 0.055 | 0.02 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.090; 2% (8 img) gap 0.071; 3% (11 img) gap 0.042; 4% (15 img) gap 0.073; 5% (19 img) gap 0.018

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.545 | 0.490 | 0.545 | 0.421 |
| continue | 60 | 0.562 | 0.488 | 0.537 | 0.527 |
| dacs | 180 | 0.711 | 0.536 | 0.691 | 0.688 |
| dann | 180 | 0.578 | 0.548 | 0.521 | 0.373 |
| fda | 180 | 0.586 | 0.535 | 0.503 | 0.562 |
| minent | 180 | 0.604 | 0.543 | 0.526 | 0.502 |
| minent_mmd | 180 | 0.621 | 0.565 | 0.194 | 0.194 |
| mmd | 180 | 0.588 | 0.504 | 0.521 | 0.366 |
| source_only | 180 | 0.534 | 0.500 | 0.383 | 0.187 |

Last-checkpoint reference: within-run gap 0.053.

## etis (n=1500)

Oracle Dice 0.479 (minent_etis_a0_seed43_l1_s0200); pool mean 0.363

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | dacs | 0.410 | 0.068 [0.028, 0.108] | 0.068 | 0.46 | 0.32 | – | 0.026 | 0.41 |
| entropy | minent | 0.365 | 0.114 [0.066, 0.161] | 0.081 | 0.70 | 0.51 | – | 0.032 | 0.26 |
| infomax | continue | 0.292 | 0.187 [0.137, 0.237] | 0.187 | -0.27 | -0.20 | – | 0.057 | -0.23 |
| bnm | continue | 0.292 | 0.187 [0.138, 0.239] | 0.187 | -0.17 | -0.14 | – | 0.055 | -0.20 |
| mcc | adabn | 0.349 | 0.129 [0.082, 0.177] | 0.129 | 0.03 | -0.00 | – | 0.050 | -0.24 |
| cmr_nhd_gauss | source_only | 0.137 | 0.341 [0.289, 0.390] | 0.109 | 0.50 | 0.35 | – | 0.038 | 0.21 |
| cmr_ei_gauss | minent | 0.360 | 0.119 [0.075, 0.165] | 0.109 | 0.56 | 0.40 | – | 0.033 | 0.25 |
| cmr_nhd_fg_gauss | source_only | 0.236 | 0.243 [0.185, 0.296] | 0.243 | -0.02 | -0.01 | – | 0.043 | -0.05 |
| cmr_ei_fg_gauss | source_only | 0.236 | 0.243 [0.188, 0.296] | 0.243 | -0.01 | -0.01 | – | 0.041 | -0.06 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (2 img) gap 0.078; 2% (4 img) gap 0.088; 3% (6 img) gap 0.073; 4% (8 img) gap 0.056; 5% (10 img) gap 0.050

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.392 | 0.337 | 0.338 | 0.267 |
| continue | 60 | 0.404 | 0.359 | 0.368 | 0.314 |
| dacs | 180 | 0.468 | 0.410 | 0.415 | 0.415 |
| dann | 180 | 0.439 | 0.393 | 0.391 | 0.377 |
| fda | 180 | 0.473 | 0.421 | 0.421 | 0.421 |
| minent | 180 | 0.479 | 0.403 | 0.365 | 0.360 |
| minent_mmd | 180 | 0.476 | 0.380 | 0.307 | 0.307 |
| mmd | 180 | 0.462 | 0.353 | 0.411 | 0.419 |
| source_only | 180 | 0.423 | 0.374 | 0.314 | 0.137 |

Last-checkpoint reference: within-run gap 0.046.

## polypgen (n=180)

Oracle Dice 0.596 (source_a1_seed44_e023); pool mean 0.479

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.596 | 0.000 [0.000, 0.000] | 0.000 | 0.96 | 0.84 | 0.308 | 0.012 | 0.96 |
| entropy | source_only | 0.551 | 0.044 [-0.013, 0.076] | 0.017 | 0.84 | 0.66 | 0.141 | 0.026 | 0.89 |
| infomax | source_only | 0.520 | 0.075 [0.056, 0.088] | 0.075 | 0.59 | 0.42 | 0.656 | 0.103 | 0.60 |
| bnm | source_only | 0.520 | 0.075 [0.056, 0.089] | 0.075 | 0.72 | 0.54 | 0.656 | 0.076 | 0.73 |
| mcc | source_only | 0.520 | 0.075 [0.055, 0.088] | 0.075 | 0.81 | 0.62 | 0.656 | 0.054 | 0.82 |
| cmr_nhd_gauss | source_only | 0.412 | 0.183 [0.111, 0.223] | 0.051 | 0.35 | 0.24 | 0.236 | 0.137 | 0.36 |
| cmr_ei_gauss | source_only | 0.551 | 0.044 [-0.012, 0.074] | 0.051 | 0.53 | 0.37 | 0.141 | 0.063 | 0.55 |
| cmr_nhd_fg_gauss | source_only | 0.381 | 0.214 [0.135, 0.272] | 0.214 | -0.23 | -0.18 | 0.823 | 0.205 | -0.26 |
| cmr_ei_fg_gauss | source_only | 0.381 | 0.214 [0.127, 0.273] | 0.214 | -0.22 | -0.17 | 0.823 | 0.205 | -0.25 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (15 img) gap 0.036; 2% (31 img) gap 0.024; 3% (46 img) gap 0.015; 4% (61 img) gap 0.025; 5% (77 img) gap 0.017

Last-checkpoint reference: within-run gap 0.018.

## far (n=180)

Oracle Dice 0.533 (source_a1_seed43_e024); pool mean 0.408

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.521 | 0.012 [-0.026, 0.042] | 0.012 | 0.92 | 0.75 | 0.597 | 0.009 | 0.94 |
| entropy | source_only | 0.491 | 0.043 [0.014, 0.076] | 0.004 | 0.87 | 0.68 | 0.295 | 0.031 | 0.88 |
| infomax | source_only | 0.370 | 0.163 [0.112, 0.195] | 0.163 | 0.18 | 0.11 | 0.950 | 0.127 | 0.20 |
| bnm | source_only | 0.418 | 0.115 [0.069, 0.154] | 0.115 | 0.47 | 0.32 | 0.797 | 0.096 | 0.51 |
| mcc | source_only | 0.479 | 0.055 [0.030, 0.111] | 0.055 | 0.56 | 0.39 | 0.714 | 0.080 | 0.59 |
| cmr_nhd_gauss | source_only | 0.432 | 0.101 [0.052, 0.139] | 0.289 | 0.04 | 0.03 | 0.407 | 0.099 | 0.04 |
| cmr_ei_gauss | source_only | 0.491 | 0.043 [0.012, 0.075] | 0.022 | 0.55 | 0.40 | 0.295 | 0.064 | 0.59 |
| cmr_nhd_fg_gauss | source_only | 0.245 | 0.289 [0.252, 0.339] | 0.289 | -0.54 | -0.41 | 0.988 | 0.251 | -0.53 |
| cmr_ei_fg_gauss | source_only | 0.245 | 0.289 [0.252, 0.341] | 0.289 | -0.52 | -0.39 | 0.988 | 0.251 | -0.52 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.038; 2% (9 img) gap 0.038; 3% (13 img) gap 0.033; 4% (17 img) gap 0.039; 5% (22 img) gap 0.031

Last-checkpoint reference: within-run gap 0.027.

