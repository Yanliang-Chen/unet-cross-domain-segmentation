# Selection gap: `clinicdb_full` (pools: source_only, uda_adabn, uda_minent, uda_advent, uda_dann, uda_mmd, uda_bnm, uda_mcc, uda_meanteacher, uda_fda, uda_continue)

## clinicdb (n=1860)

Oracle Dice 0.714 (minent_clinicdb_a0_seed44_l0.5_s2000); pool mean 0.592; validator ensemble Dice 0.688

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | continue | 0.676 | 0.038 [-0.023, 0.065] | 0.038 | 0.31 | 0.23 | – | 0.025 | 0.37 |
| iwcv | mcc | 0.445 | 0.269 [0.172, 0.333] | 0.269 | -0.18 | -0.13 | – | 0.044 | -0.09 |
| dev | mmd | 0.623 | 0.091 [0.004, 0.121] | 0.091 | 0.16 | 0.10 | – | 0.039 | 0.06 |
| devn | mmd | 0.630 | 0.084 [0.027, 0.116] | 0.084 | 0.16 | 0.11 | – | 0.040 | 0.07 |
| entropy | mcc | 0.450 | 0.264 [0.172, 0.338] | 0.264 | 0.45 | 0.35 | – | 0.036 | 0.22 |
| infomax | mcc | 0.408 | 0.306 [0.178, 0.391] | 0.306 | -0.35 | -0.29 | – | 0.064 | -0.32 |
| bnm | mcc | 0.408 | 0.306 [0.177, 0.398] | 0.306 | -0.32 | -0.26 | – | 0.063 | -0.30 |
| mcc | mcc | 0.450 | 0.264 [0.169, 0.337] | 0.264 | -0.05 | -0.03 | – | 0.055 | -0.19 |
| snd | mmd | 0.599 | 0.115 [0.052, 0.158] | 0.115 | 0.19 | 0.13 | – | 0.048 | -0.17 |
| mixval | source_only | 0.115 | 0.599 [0.472, 0.652] | 0.043 | 0.12 | 0.08 | – | 0.035 | 0.15 |
| mixval_fg | mcc | 0.415 | 0.299 [0.185, 0.355] | 0.299 | -0.09 | -0.06 | – | 0.045 | 0.02 |
| cmr_nhd_gauss | source_only | 0.146 | 0.568 [0.427, 0.628] | 0.070 | 0.40 | 0.27 | – | 0.038 | 0.26 |
| cmr_ei_gauss | minent | 0.643 | 0.070 [0.028, 0.113] | 0.070 | 0.45 | 0.32 | – | 0.032 | 0.29 |
| cmr_nhd_fg_gauss | meanteacher | 0.660 | 0.054 [-0.034, 0.088] | 0.054 | 0.27 | 0.19 | – | 0.037 | 0.17 |
| cmr_ei_fg_gauss | meanteacher | 0.641 | 0.073 [-0.011, 0.124] | 0.073 | 0.27 | 0.19 | – | 0.034 | 0.16 |
| cmr_nhd_ttd | source_only | 0.146 | 0.568 [0.428, 0.625] | 0.046 | 0.18 | 0.12 | – | 0.042 | 0.26 |
| cmr_ei_ttd | minent | 0.651 | 0.063 [-0.028, 0.101] | 0.063 | 0.41 | 0.32 | – | 0.028 | 0.26 |
| cmr_nhd_fg_ttd | minent | 0.662 | 0.052 [-0.023, 0.090] | 0.052 | 0.09 | 0.06 | – | 0.035 | 0.21 |
| cmr_ei_fg_ttd | mcc | 0.415 | 0.299 [0.205, 0.366] | 0.299 | 0.34 | 0.27 | – | 0.032 | 0.20 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.077; 2% (9 img) gap 0.056; 3% (13 img) gap 0.046; 4% (17 img) gap 0.019; 5% (21 img) gap 0.034

| Algorithm | n | Oracle Dice | Source-Risk pick | Entropy pick | CMR-EI pick |
|---|---:|---:|---:|---:|---:|
| adabn | 180 | 0.660 | 0.602 | 0.660 | 0.559 |
| advent | 180 | 0.696 | 0.670 | 0.668 | 0.668 |
| bnm | 180 | 0.696 | 0.667 | 0.661 | 0.653 |
| continue | 60 | 0.696 | 0.676 | 0.676 | 0.655 |
| dann | 180 | 0.695 | 0.669 | 0.665 | 0.643 |
| fda | 180 | 0.697 | 0.668 | 0.648 | 0.677 |
| mcc | 180 | 0.681 | 0.548 | 0.450 | 0.681 |
| meanteacher | 180 | 0.673 | 0.570 | 0.581 | 0.641 |
| minent | 180 | 0.714 | 0.648 | 0.675 | 0.643 |
| mmd | 180 | 0.707 | 0.635 | 0.645 | 0.639 |
| source_only | 180 | 0.684 | 0.684 | 0.622 | 0.575 |

Last-checkpoint reference: within-run gap 0.043.

## colondb (n=180)

Oracle Dice 0.534 (source_a1_seed44_e018); pool mean 0.342

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.500 | 0.034 [0.011, 0.057] | 0.034 | 0.87 | 0.68 | – | 0.058 | 0.86 |
| entropy | source_only | 0.383 | 0.151 [0.122, 0.183] | 0.081 | 0.73 | 0.53 | – | 0.086 | 0.79 |
| infomax | source_only | 0.370 | 0.164 [0.132, 0.197] | 0.164 | 0.67 | 0.49 | – | 0.121 | 0.67 |
| bnm | source_only | 0.370 | 0.164 [0.133, 0.196] | 0.164 | 0.76 | 0.57 | – | 0.112 | 0.77 |
| mcc | source_only | 0.370 | 0.164 [0.132, 0.196] | 0.164 | 0.81 | 0.63 | – | 0.101 | 0.83 |
| cmr_nhd_gauss | source_only | 0.043 | 0.492 [0.453, 0.529] | 0.122 | 0.08 | 0.05 | – | 0.244 | 0.07 |
| cmr_ei_gauss | source_only | 0.187 | 0.347 [0.307, 0.388] | 0.122 | 0.32 | 0.22 | – | 0.192 | 0.33 |
| cmr_nhd_fg_gauss | source_only | 0.293 | 0.241 [0.206, 0.275] | 0.241 | 0.17 | 0.13 | – | 0.178 | 0.14 |
| cmr_ei_fg_gauss | source_only | 0.293 | 0.241 [0.205, 0.276] | 0.241 | 0.18 | 0.13 | – | 0.178 | 0.15 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.061; 2% (8 img) gap 0.059; 3% (11 img) gap 0.044; 4% (15 img) gap 0.066; 5% (19 img) gap 0.050

Last-checkpoint reference: within-run gap 0.059.

## etis (n=180)

Oracle Dice 0.423 (source_a0_seed42_e018); pool mean 0.300

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.374 | 0.050 [0.016, 0.086] | 0.050 | 0.82 | 0.62 | – | 0.033 | 0.81 |
| entropy | source_only | 0.314 | 0.110 [0.075, 0.148] | 0.025 | 0.68 | 0.50 | – | 0.077 | 0.73 |
| infomax | source_only | 0.303 | 0.120 [0.077, 0.161] | 0.120 | 0.55 | 0.39 | – | 0.126 | 0.56 |
| bnm | source_only | 0.303 | 0.120 [0.078, 0.161] | 0.120 | 0.69 | 0.51 | – | 0.099 | 0.69 |
| mcc | source_only | 0.299 | 0.124 [0.084, 0.164] | 0.124 | 0.75 | 0.55 | – | 0.046 | 0.75 |
| cmr_nhd_gauss | source_only | 0.137 | 0.286 [0.240, 0.331] | 0.063 | 0.22 | 0.14 | – | 0.163 | 0.29 |
| cmr_ei_gauss | source_only | 0.137 | 0.286 [0.242, 0.332] | 0.063 | 0.41 | 0.28 | – | 0.153 | 0.47 |
| cmr_nhd_fg_gauss | source_only | 0.236 | 0.187 [0.135, 0.239] | 0.187 | 0.11 | 0.08 | – | 0.097 | 0.14 |
| cmr_ei_fg_gauss | source_only | 0.236 | 0.187 [0.138, 0.236] | 0.187 | 0.12 | 0.09 | – | 0.097 | 0.15 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (2 img) gap 0.074; 2% (4 img) gap 0.054; 3% (6 img) gap 0.044; 4% (8 img) gap 0.055; 5% (10 img) gap 0.062

Last-checkpoint reference: within-run gap 0.047.

## polypgen (n=180)

Oracle Dice 0.596 (source_a1_seed44_e023); pool mean 0.479

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.596 | 0.000 [0.000, 0.000] | 0.000 | 0.96 | 0.84 | 0.308 | 0.012 | 0.96 |
| entropy | source_only | 0.551 | 0.044 [-0.010, 0.075] | 0.017 | 0.84 | 0.66 | 0.141 | 0.026 | 0.89 |
| infomax | source_only | 0.520 | 0.075 [0.056, 0.089] | 0.075 | 0.59 | 0.42 | 0.656 | 0.103 | 0.60 |
| bnm | source_only | 0.520 | 0.075 [0.056, 0.089] | 0.075 | 0.72 | 0.54 | 0.656 | 0.076 | 0.73 |
| mcc | source_only | 0.520 | 0.075 [0.056, 0.088] | 0.075 | 0.81 | 0.62 | 0.656 | 0.054 | 0.82 |
| cmr_nhd_gauss | source_only | 0.412 | 0.183 [0.111, 0.224] | 0.051 | 0.35 | 0.24 | 0.236 | 0.137 | 0.36 |
| cmr_ei_gauss | source_only | 0.551 | 0.044 [-0.012, 0.074] | 0.051 | 0.53 | 0.37 | 0.141 | 0.063 | 0.55 |
| cmr_nhd_fg_gauss | source_only | 0.381 | 0.214 [0.125, 0.272] | 0.214 | -0.23 | -0.18 | 0.823 | 0.205 | -0.26 |
| cmr_ei_fg_gauss | source_only | 0.381 | 0.214 [0.131, 0.273] | 0.214 | -0.22 | -0.17 | 0.823 | 0.205 | -0.25 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (15 img) gap 0.029; 2% (31 img) gap 0.009; 3% (46 img) gap 0.014; 4% (61 img) gap 0.015; 5% (77 img) gap 0.022

Last-checkpoint reference: within-run gap 0.018.

## far (n=180)

Oracle Dice 0.533 (source_a1_seed43_e024); pool mean 0.408

| Validator | Pick | Pick Dice | Gap [95% CI] | Guarded gap | ρ | τ | Pick FPR | Within-run gap | Within-run ρ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| source_risk | source_only | 0.521 | 0.012 [-0.027, 0.041] | 0.012 | 0.92 | 0.75 | 0.597 | 0.009 | 0.94 |
| entropy | source_only | 0.491 | 0.043 [0.014, 0.076] | 0.004 | 0.87 | 0.68 | 0.295 | 0.031 | 0.88 |
| infomax | source_only | 0.370 | 0.163 [0.112, 0.195] | 0.163 | 0.18 | 0.11 | 0.950 | 0.127 | 0.20 |
| bnm | source_only | 0.418 | 0.115 [0.069, 0.155] | 0.115 | 0.47 | 0.32 | 0.797 | 0.096 | 0.51 |
| mcc | source_only | 0.479 | 0.055 [0.028, 0.113] | 0.055 | 0.56 | 0.39 | 0.714 | 0.080 | 0.59 |
| cmr_nhd_gauss | source_only | 0.432 | 0.101 [0.053, 0.138] | 0.289 | 0.04 | 0.03 | 0.407 | 0.099 | 0.04 |
| cmr_ei_gauss | source_only | 0.491 | 0.043 [0.014, 0.076] | 0.022 | 0.55 | 0.40 | 0.295 | 0.064 | 0.59 |
| cmr_nhd_fg_gauss | source_only | 0.245 | 0.289 [0.252, 0.339] | 0.289 | -0.54 | -0.41 | 0.988 | 0.251 | -0.53 |
| cmr_ei_fg_gauss | source_only | 0.245 | 0.289 [0.254, 0.339] | 0.289 | -0.52 | -0.39 | 0.988 | 0.251 | -0.52 |

Label budget (pick by Dice on labelled subset, gap on remaining images): 1% (4 img) gap 0.062; 2% (9 img) gap 0.038; 3% (13 img) gap 0.047; 4% (17 img) gap 0.031; 5% (22 img) gap 0.021

Last-checkpoint reference: within-run gap 0.027.

