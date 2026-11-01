# Selection gap: `clinicdb_methods` (pools: source_only, uda_adabn, uda_continue, uda_minent_mmd, uda_dacs)

## clinicdb (n=780)

Oracle Dice 0.727 (dacs_clinicdb_a1_seed44_l1_s0600); pool mean 0.599

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.676 | 0.051 [-0.020, 0.099] | 0.051 | 0.81 | 0.62 | – | 0.020 | 0.51 |
| entropy | minent_mmd | 0.610 | 0.118 [0.062, 0.160] | 0.118 | 0.72 | 0.52 | – | 0.041 | 0.19 |
| infomax | continue | 0.520 | 0.207 [0.114, 0.264] | 0.207 | 0.34 | 0.24 | – | 0.047 | 0.14 |
| bnm | continue | 0.520 | 0.207 [0.111, 0.272] | 0.207 | 0.38 | 0.27 | – | 0.047 | 0.17 |
| mcc | dacs | 0.726 | 0.001 [-0.028, 0.023] | 0.001 | 0.63 | 0.46 | – | 0.045 | -0.02 |
| cmr_nhd_gauss | source_only | 0.146 | 0.581 [0.448, 0.639] | 0.085 | 0.45 | 0.31 | – | 0.050 | 0.23 |
| cmr_ei_gauss | minent_mmd | 0.642 | 0.085 [-0.006, 0.128] | 0.085 | 0.57 | 0.40 | – | 0.033 | 0.29 |
| cmr_nhd_fg_gauss | dacs | 0.670 | 0.057 [-0.038, 0.087] | 0.057 | 0.53 | 0.37 | – | 0.042 | 0.20 |
| cmr_ei_fg_gauss | minent_mmd | 0.590 | 0.137 [0.057, 0.199] | 0.137 | 0.55 | 0.39 | – | 0.038 | 0.20 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.061; 2% (9 img) gap 0.055; 3% (13 img) gap 0.038; 4% (17 img) gap 0.033; 5% (21 img) gap 0.045

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.660 | 0.602 | 0.660 | 0.559 |
| continue | 60 | 0.696 | 0.676 | 0.676 | 0.655 |
| dacs | 180 | 0.727 | 0.698 | 0.726 | 0.670 |
| minent_mmd | 180 | 0.713 | 0.663 | 0.610 | 0.642 |
| source_only | 180 | 0.684 | 0.684 | 0.622 | 0.575 |

Last-checkpoint reference: within-run gap 0.044.

## colondb (n=180)

Oracle Dice 0.534 (source_a1_seed44_e018); pool mean 0.342

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.500 | 0.034 [0.012, 0.057] | 0.034 | 0.87 | 0.68 | – | 0.058 | 0.86 |
| entropy | source_only | 0.383 | 0.151 [0.120, 0.184] | 0.081 | 0.73 | 0.53 | – | 0.086 | 0.79 |
| infomax | source_only | 0.370 | 0.164 [0.134, 0.197] | 0.164 | 0.67 | 0.49 | – | 0.121 | 0.67 |
| bnm | source_only | 0.370 | 0.164 [0.132, 0.197] | 0.164 | 0.76 | 0.57 | – | 0.112 | 0.77 |
| mcc | source_only | 0.370 | 0.164 [0.133, 0.197] | 0.164 | 0.81 | 0.63 | – | 0.101 | 0.83 |
| cmr_nhd_gauss | source_only | 0.043 | 0.492 [0.453, 0.529] | 0.122 | 0.08 | 0.05 | – | 0.244 | 0.07 |
| cmr_ei_gauss | source_only | 0.187 | 0.347 [0.308, 0.387] | 0.122 | 0.32 | 0.22 | – | 0.192 | 0.33 |
| cmr_nhd_fg_gauss | source_only | 0.293 | 0.241 [0.208, 0.274] | 0.241 | 0.17 | 0.13 | – | 0.178 | 0.14 |
| cmr_ei_fg_gauss | source_only | 0.293 | 0.241 [0.209, 0.276] | 0.241 | 0.18 | 0.13 | – | 0.178 | 0.15 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.087; 2% (8 img) gap 0.042; 3% (11 img) gap 0.049; 4% (15 img) gap 0.044; 5% (19 img) gap 0.049

Last-checkpoint reference: within-run gap 0.059.

## etis (n=180)

Oracle Dice 0.423 (source_a0_seed42_e018); pool mean 0.300

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.374 | 0.050 [0.015, 0.083] | 0.050 | 0.82 | 0.62 | – | 0.033 | 0.81 |
| entropy | source_only | 0.314 | 0.110 [0.075, 0.148] | 0.025 | 0.68 | 0.50 | – | 0.077 | 0.73 |
| infomax | source_only | 0.303 | 0.120 [0.080, 0.161] | 0.120 | 0.55 | 0.39 | – | 0.126 | 0.56 |
| bnm | source_only | 0.303 | 0.120 [0.078, 0.163] | 0.120 | 0.69 | 0.51 | – | 0.099 | 0.69 |
| mcc | source_only | 0.299 | 0.124 [0.085, 0.164] | 0.124 | 0.75 | 0.55 | – | 0.046 | 0.75 |
| cmr_nhd_gauss | source_only | 0.137 | 0.286 [0.240, 0.332] | 0.063 | 0.22 | 0.14 | – | 0.163 | 0.29 |
| cmr_ei_gauss | source_only | 0.137 | 0.286 [0.239, 0.334] | 0.063 | 0.41 | 0.28 | – | 0.153 | 0.47 |
| cmr_nhd_fg_gauss | source_only | 0.236 | 0.187 [0.138, 0.237] | 0.187 | 0.11 | 0.08 | – | 0.097 | 0.14 |
| cmr_ei_fg_gauss | source_only | 0.236 | 0.187 [0.136, 0.234] | 0.187 | 0.12 | 0.09 | – | 0.097 | 0.15 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (2 img) gap 0.087; 2% (4 img) gap 0.079; 3% (6 img) gap 0.042; 4% (8 img) gap 0.059; 5% (10 img) gap 0.053

Last-checkpoint reference: within-run gap 0.047.

## polypgen (n=180)

Oracle Dice 0.596 (source_a1_seed44_e023); pool mean 0.479

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.596 | 0.000 [0.000, 0.000] | 0.000 | 0.96 | 0.84 | 0.308 | 0.012 | 0.96 |
| entropy | source_only | 0.551 | 0.044 [-0.012, 0.077] | 0.017 | 0.84 | 0.66 | 0.141 | 0.026 | 0.89 |
| infomax | source_only | 0.520 | 0.075 [0.057, 0.088] | 0.075 | 0.59 | 0.42 | 0.656 | 0.103 | 0.60 |
| bnm | source_only | 0.520 | 0.075 [0.055, 0.088] | 0.075 | 0.72 | 0.54 | 0.656 | 0.076 | 0.73 |
| mcc | source_only | 0.520 | 0.075 [0.056, 0.089] | 0.075 | 0.81 | 0.62 | 0.656 | 0.054 | 0.82 |
| cmr_nhd_gauss | source_only | 0.412 | 0.183 [0.110, 0.223] | 0.051 | 0.35 | 0.24 | 0.236 | 0.137 | 0.36 |
| cmr_ei_gauss | source_only | 0.551 | 0.044 [-0.009, 0.074] | 0.051 | 0.53 | 0.37 | 0.141 | 0.063 | 0.55 |
| cmr_nhd_fg_gauss | source_only | 0.381 | 0.214 [0.135, 0.276] | 0.214 | -0.23 | -0.18 | 0.823 | 0.205 | -0.26 |
| cmr_ei_fg_gauss | source_only | 0.381 | 0.214 [0.127, 0.273] | 0.214 | -0.22 | -0.17 | 0.823 | 0.205 | -0.25 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (15 img) gap 0.039; 2% (31 img) gap 0.042; 3% (46 img) gap 0.021; 4% (61 img) gap 0.021; 5% (77 img) gap 0.013

Last-checkpoint reference: within-run gap 0.018.

## far (n=180)

Oracle Dice 0.533 (source_a1_seed43_e024); pool mean 0.408

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.521 | 0.012 [-0.026, 0.042] | 0.012 | 0.92 | 0.75 | 0.597 | 0.009 | 0.94 |
| entropy | source_only | 0.491 | 0.043 [0.013, 0.077] | 0.004 | 0.87 | 0.68 | 0.295 | 0.031 | 0.88 |
| infomax | source_only | 0.370 | 0.163 [0.111, 0.195] | 0.163 | 0.18 | 0.11 | 0.950 | 0.127 | 0.20 |
| bnm | source_only | 0.418 | 0.115 [0.071, 0.156] | 0.115 | 0.47 | 0.32 | 0.797 | 0.096 | 0.51 |
| mcc | source_only | 0.479 | 0.055 [0.029, 0.108] | 0.055 | 0.56 | 0.39 | 0.714 | 0.080 | 0.59 |
| cmr_nhd_gauss | source_only | 0.432 | 0.101 [0.054, 0.139] | 0.289 | 0.04 | 0.03 | 0.407 | 0.099 | 0.04 |
| cmr_ei_gauss | source_only | 0.491 | 0.043 [0.015, 0.074] | 0.022 | 0.55 | 0.40 | 0.295 | 0.064 | 0.59 |
| cmr_nhd_fg_gauss | source_only | 0.245 | 0.289 [0.252, 0.340] | 0.289 | -0.54 | -0.41 | 0.988 | 0.251 | -0.53 |
| cmr_ei_fg_gauss | source_only | 0.245 | 0.289 [0.254, 0.342] | 0.289 | -0.52 | -0.39 | 0.988 | 0.251 | -0.52 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.037; 2% (9 img) gap 0.035; 3% (13 img) gap 0.037; 4% (17 img) gap 0.039; 5% (22 img) gap 0.042

Last-checkpoint reference: within-run gap 0.027.

