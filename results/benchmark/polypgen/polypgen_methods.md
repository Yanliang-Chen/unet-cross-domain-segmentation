# Selection gap: `polypgen_methods` (pools: source_only, uda_adabn, uda_continue, uda_dacs, uda_minent_mmd, uda_fda)

## clinicdb (n=960)

Oracle Dice 0.727 (dacs_clinicdb_a1_seed44_l1_s0600); pool mean 0.608

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.676 | 0.051 [-0.020, 0.099] | 0.051 | 0.78 | 0.60 | – | 0.020 | 0.48 |
| entropy | minent_mmd | 0.610 | 0.118 [0.062, 0.160] | 0.118 | 0.66 | 0.47 | – | 0.036 | 0.23 |
| infomax | continue | 0.520 | 0.207 [0.114, 0.264] | 0.207 | 0.28 | 0.20 | – | 0.045 | 0.06 |
| bnm | continue | 0.520 | 0.207 [0.111, 0.272] | 0.207 | 0.32 | 0.23 | – | 0.043 | 0.09 |
| mcc | dacs | 0.726 | 0.001 [-0.028, 0.023] | 0.001 | 0.56 | 0.41 | – | 0.040 | 0.03 |
| cmr_nhd_gauss | source_only | 0.146 | 0.581 [0.448, 0.639] | 0.085 | 0.41 | 0.28 | – | 0.043 | 0.19 |
| cmr_ei_gauss | minent_mmd | 0.642 | 0.085 [-0.006, 0.128] | 0.085 | 0.51 | 0.36 | – | 0.030 | 0.24 |
| cmr_nhd_fg_gauss | dacs | 0.670 | 0.057 [-0.038, 0.087] | 0.057 | 0.51 | 0.36 | – | 0.037 | 0.17 |
| cmr_ei_fg_gauss | minent_mmd | 0.590 | 0.137 [0.057, 0.199] | 0.137 | 0.53 | 0.37 | – | 0.033 | 0.17 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.061; 2% (9 img) gap 0.057; 3% (13 img) gap 0.048; 4% (17 img) gap 0.033; 5% (21 img) gap 0.048

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.660 | 0.602 | 0.660 | 0.559 |
| continue | 60 | 0.696 | 0.676 | 0.676 | 0.655 |
| dacs | 180 | 0.727 | 0.698 | 0.726 | 0.670 |
| fda | 180 | 0.697 | 0.668 | 0.648 | 0.677 |
| minent_mmd | 180 | 0.713 | 0.663 | 0.610 | 0.642 |
| source_only | 180 | 0.684 | 0.684 | 0.622 | 0.575 |

Last-checkpoint reference: within-run gap 0.040.

## colondb (n=960)

Oracle Dice 0.711 (dacs_colondb_a1_seed44_l1_s1800); pool mean 0.475

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.488 | 0.224 [0.195, 0.252] | 0.224 | 0.74 | 0.56 | – | 0.033 | 0.27 |
| entropy | minent_mmd | 0.194 | 0.517 [0.482, 0.554] | 0.132 | 0.64 | 0.46 | – | 0.049 | 0.18 |
| infomax | continue | 0.297 | 0.415 [0.383, 0.447] | 0.415 | 0.16 | 0.14 | – | 0.051 | 0.22 |
| bnm | continue | 0.297 | 0.415 [0.382, 0.447] | 0.415 | 0.25 | 0.21 | – | 0.047 | 0.26 |
| mcc | minent_mmd | 0.404 | 0.307 [0.274, 0.340] | 0.307 | 0.48 | 0.36 | – | 0.046 | 0.20 |
| cmr_nhd_gauss | source_only | 0.043 | 0.668 [0.636, 0.699] | 0.299 | 0.17 | 0.12 | – | 0.075 | 0.03 |
| cmr_ei_gauss | minent_mmd | 0.194 | 0.517 [0.483, 0.554] | 0.299 | 0.36 | 0.26 | – | 0.061 | 0.11 |
| cmr_nhd_fg_gauss | adabn | 0.291 | 0.421 [0.390, 0.450] | 0.421 | 0.04 | 0.03 | – | 0.059 | 0.09 |
| cmr_ei_fg_gauss | adabn | 0.291 | 0.421 [0.390, 0.451] | 0.421 | 0.05 | 0.04 | – | 0.058 | 0.09 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.075; 2% (8 img) gap 0.031; 3% (11 img) gap 0.042; 4% (15 img) gap 0.029; 5% (19 img) gap 0.028

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.545 | 0.490 | 0.545 | 0.421 |
| continue | 60 | 0.562 | 0.488 | 0.537 | 0.527 |
| dacs | 180 | 0.711 | 0.536 | 0.691 | 0.688 |
| fda | 180 | 0.586 | 0.535 | 0.503 | 0.562 |
| minent_mmd | 180 | 0.621 | 0.565 | 0.194 | 0.194 |
| source_only | 180 | 0.534 | 0.500 | 0.383 | 0.187 |

Last-checkpoint reference: within-run gap 0.047.

## etis (n=960)

Oracle Dice 0.476 (minent_mmd_etis_a0_seed43_l1_s0800); pool mean 0.359

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | dacs | 0.410 | 0.065 [0.022, 0.107] | 0.065 | 0.63 | 0.45 | – | 0.024 | 0.45 |
| entropy | minent_mmd | 0.307 | 0.168 [0.122, 0.216] | 0.089 | 0.77 | 0.58 | – | 0.029 | 0.33 |
| infomax | continue | 0.292 | 0.184 [0.139, 0.228] | 0.184 | -0.09 | -0.08 | – | 0.052 | -0.08 |
| bnm | continue | 0.292 | 0.184 [0.138, 0.228] | 0.184 | 0.03 | 0.01 | – | 0.048 | -0.03 |
| mcc | adabn | 0.349 | 0.126 [0.082, 0.169] | 0.126 | 0.29 | 0.19 | – | 0.038 | -0.02 |
| cmr_nhd_gauss | source_only | 0.137 | 0.338 [0.288, 0.389] | 0.060 | 0.47 | 0.33 | – | 0.045 | 0.21 |
| cmr_ei_gauss | source_only | 0.137 | 0.338 [0.289, 0.388] | 0.060 | 0.56 | 0.41 | – | 0.039 | 0.26 |
| cmr_nhd_fg_gauss | source_only | 0.236 | 0.239 [0.190, 0.291] | 0.239 | 0.05 | 0.04 | – | 0.045 | -0.05 |
| cmr_ei_fg_gauss | source_only | 0.236 | 0.239 [0.189, 0.290] | 0.239 | 0.05 | 0.04 | – | 0.044 | -0.06 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (2 img) gap 0.084; 2% (4 img) gap 0.060; 3% (6 img) gap 0.054; 4% (8 img) gap 0.052; 5% (10 img) gap 0.061

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.392 | 0.337 | 0.338 | 0.267 |
| continue | 60 | 0.404 | 0.359 | 0.368 | 0.314 |
| dacs | 180 | 0.468 | 0.410 | 0.415 | 0.415 |
| fda | 180 | 0.473 | 0.421 | 0.421 | 0.421 |
| minent_mmd | 180 | 0.476 | 0.380 | 0.307 | 0.307 |
| source_only | 180 | 0.423 | 0.374 | 0.314 | 0.137 |

Last-checkpoint reference: within-run gap 0.042.

## polypgen (n=660)

Oracle Dice 0.632 (fda_polypgen_a0_seed43_l0.1_s1200); pool mean 0.514

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | dacs | 0.620 | 0.012 [-0.010, 0.028] | 0.012 | 0.92 | 0.77 | 0.235 | 0.007 | 0.56 |
| entropy | minent_mmd | 0.515 | 0.117 [0.053, 0.157] | 0.045 | 0.84 | 0.64 | 0.047 | 0.016 | 0.26 |
| infomax | adabn | 0.496 | 0.136 [0.090, 0.185] | 0.136 | -0.02 | -0.02 | 0.912 | 0.031 | -0.06 |
| bnm | adabn | 0.496 | 0.136 [0.090, 0.186] | 0.136 | 0.10 | 0.07 | 0.912 | 0.027 | -0.04 |
| mcc | minent_mmd | 0.583 | 0.050 [0.026, 0.092] | 0.050 | 0.42 | 0.29 | 0.908 | 0.025 | -0.05 |
| cmr_nhd_gauss | source_only | 0.412 | 0.220 [0.139, 0.265] | 0.015 | 0.24 | 0.17 | 0.236 | 0.037 | 0.01 |
| cmr_ei_gauss | minent_mmd | 0.515 | 0.117 [0.052, 0.158] | 0.015 | 0.41 | 0.29 | 0.047 | 0.024 | 0.09 |
| cmr_nhd_fg_gauss | source_only | 0.381 | 0.251 [0.162, 0.321] | 0.251 | -0.36 | -0.26 | 0.823 | 0.044 | -0.21 |
| cmr_ei_fg_gauss | source_only | 0.381 | 0.251 [0.162, 0.317] | 0.251 | -0.35 | -0.25 | 0.823 | 0.044 | -0.20 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (15 img) gap 0.034; 2% (31 img) gap 0.017; 3% (46 img) gap 0.023; 4% (61 img) gap 0.018; 5% (77 img) gap 0.020

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.515 | 0.514 | 0.482 | 0.466 |
| continue | 30 | 0.627 | 0.597 | 0.576 | 0.577 |
| dacs | 90 | 0.626 | 0.620 | 0.597 | 0.617 |
| fda | 90 | 0.632 | 0.631 | 0.608 | 0.608 |
| minent_mmd | 90 | 0.623 | 0.610 | 0.515 | 0.515 |
| source_only | 180 | 0.596 | 0.596 | 0.551 | 0.551 |

Last-checkpoint reference: within-run gap 0.025.

## far (n=480)

Oracle Dice 0.592 (fda_far_a1_seed43_l0.5_s2000); pool mean 0.425

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.539 | 0.053 [0.003, 0.060] | 0.053 | 0.94 | 0.78 | 0.656 | 0.011 | 0.66 |
| entropy | minent_mmd | 0.555 | 0.036 [-0.018, 0.050] | 0.080 | 0.87 | 0.69 | 0.471 | 0.014 | 0.61 |
| infomax | adabn | 0.428 | 0.164 [0.104, 0.184] | 0.164 | -0.19 | -0.14 | 0.909 | 0.060 | -0.26 |
| bnm | adabn | 0.394 | 0.198 [0.161, 0.234] | 0.198 | 0.03 | -0.01 | 0.952 | 0.056 | -0.19 |
| mcc | adabn | 0.394 | 0.198 [0.163, 0.234] | 0.198 | 0.24 | 0.15 | 0.952 | 0.049 | -0.15 |
| cmr_nhd_gauss | source_only | 0.432 | 0.160 [0.104, 0.195] | 0.347 | -0.08 | -0.07 | 0.407 | 0.038 | 0.05 |
| cmr_ei_gauss | dacs | 0.564 | 0.028 [-0.018, 0.054] | 0.080 | 0.33 | 0.22 | 0.344 | 0.031 | 0.23 |
| cmr_nhd_fg_gauss | source_only | 0.245 | 0.347 [0.310, 0.390] | 0.347 | -0.68 | -0.51 | 0.988 | 0.082 | -0.42 |
| cmr_ei_fg_gauss | source_only | 0.245 | 0.347 [0.310, 0.389] | 0.347 | -0.65 | -0.48 | 0.988 | 0.079 | -0.37 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.069; 2% (9 img) gap 0.051; 3% (13 img) gap 0.041; 4% (17 img) gap 0.040; 5% (22 img) gap 0.039

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.462 | 0.441 | 0.388 | 0.381 |
| continue | 30 | 0.539 | 0.539 | 0.505 | 0.437 |
| dacs | 30 | 0.579 | 0.570 | 0.564 | 0.564 |
| fda | 30 | 0.592 | 0.587 | 0.566 | 0.543 |
| minent_mmd | 30 | 0.570 | 0.545 | 0.555 | 0.550 |
| source_only | 180 | 0.533 | 0.521 | 0.491 | 0.491 |

Last-checkpoint reference: within-run gap 0.032.

