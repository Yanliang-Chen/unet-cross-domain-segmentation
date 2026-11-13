# Label-free checkpoint selection (H4, amendment v2)

Positive-polyp Dice (v2 metrics), image means. H4 is a supporting analysis: it explains how the deployed checkpoint is chosen without target labels and where that choice fails.

## 1. The deployment rule: Source-Risk inside each trajectory

Each trajectory is one method × base run × λ with its saved checkpoints (SourceOnly: the 30 epochs of a source run). Pick = the checkpoint with the highest source-validation Dice; best = the best saved checkpoint by target Dice (needs labels); regret = best − pick. Failures = trajectories with regret ≥ 0.05 Dice, a descriptive threshold set after the curves analysis (about the size of a typical method gain), not a test. Checkpoint spacing differs by target (every 200 steps on ClinicDB, ETIS, ColonDB; every 400 for the methods on PolypGen and FAR), so regret is not comparable across spacings.

| Target | Method | Trajectories | Pick Dice | Best Dice | Mean regret | Max regret | Failures |
|---|---|---:|---:|---:|---:|---:|---:|
| ClinicDB | DACS | 18 | 0.686 | 0.698 | 0.012 | 0.029 | 0/18 |
| ClinicDB | DACS without paste | 6 | 0.700 | 0.715 | 0.015 | 0.043 | 0/6 |
| ClinicDB | DACS without confidence weight | 6 | 0.690 | 0.705 | 0.014 | 0.031 | 0/6 |
| ClinicDB | MinEnt+MMD | 18 | 0.670 | 0.684 | 0.014 | 0.040 | 0/18 |
| ClinicDB | FDA | 18 | 0.650 | 0.670 | 0.020 | 0.060 | 1/18 |
| ClinicDB | MinEnt | 18 | 0.664 | 0.693 | 0.029 | 0.079 | 2/18 |
| ClinicDB | MMD | 18 | 0.658 | 0.671 | 0.013 | 0.043 | 0/18 |
| ClinicDB | DANN | 18 | 0.649 | 0.659 | 0.010 | 0.035 | 0/18 |
| ClinicDB | AdvEnt | 18 | 0.635 | 0.653 | 0.019 | 0.057 | 1/18 |
| ClinicDB | BNM | 18 | 0.627 | 0.647 | 0.021 | 0.069 | 2/18 |
| ClinicDB | MCC | 18 | 0.476 | 0.532 | 0.055 | 0.161 | 9/18 |
| ClinicDB | Mean Teacher | 18 | 0.591 | 0.618 | 0.027 | 0.097 | 3/18 |
| ClinicDB | AdaBN | 6 | 0.606 | 0.648 | 0.042 | 0.072 | 3/6 |
| ClinicDB | Source-only continuation | 6 | 0.654 | 0.677 | 0.023 | 0.050 | 0/6 |
| ClinicDB | Control | 6 | 0.637 | 0.651 | 0.013 | 0.025 | 0/6 |
| ClinicDB | SourceOnly | 6 | 0.610 | 0.660 | 0.050 | 0.110 | 3/6 |
| ETIS | DACS | 18 | 0.403 | 0.426 | 0.022 | 0.077 | 3/18 |
| ETIS | DACS without paste | 6 | 0.431 | 0.447 | 0.016 | 0.034 | 0/6 |
| ETIS | MinEnt+MMD | 18 | 0.412 | 0.432 | 0.020 | 0.072 | 3/18 |
| ETIS | FDA | 18 | 0.430 | 0.446 | 0.016 | 0.051 | 1/18 |
| ETIS | MinEnt | 18 | 0.393 | 0.427 | 0.035 | 0.067 | 5/18 |
| ETIS | MMD | 18 | 0.388 | 0.412 | 0.024 | 0.088 | 2/18 |
| ETIS | DANN | 18 | 0.362 | 0.393 | 0.031 | 0.088 | 5/18 |
| ETIS | AdaBN | 6 | 0.334 | 0.368 | 0.034 | 0.053 | 1/6 |
| ETIS | Source-only continuation | 6 | 0.379 | 0.406 | 0.026 | 0.038 | 0/6 |
| ETIS | Control | 6 | 0.338 | 0.379 | 0.041 | 0.093 | 2/6 |
| ETIS | SourceOnly | 6 | 0.366 | 0.398 | 0.033 | 0.062 | 2/6 |
| ColonDB | DACS | 18 | 0.609 | 0.635 | 0.025 | 0.050 | 1/18 |
| ColonDB | DACS without paste | 6 | 0.580 | 0.601 | 0.021 | 0.058 | 1/6 |
| ColonDB | MinEnt+MMD | 18 | 0.568 | 0.584 | 0.016 | 0.038 | 0/18 |
| ColonDB | FDA | 18 | 0.508 | 0.540 | 0.032 | 0.065 | 4/18 |
| ColonDB | MinEnt | 18 | 0.534 | 0.560 | 0.027 | 0.056 | 2/18 |
| ColonDB | MMD | 18 | 0.541 | 0.559 | 0.018 | 0.049 | 0/18 |
| ColonDB | DANN | 18 | 0.498 | 0.522 | 0.025 | 0.102 | 5/18 |
| ColonDB | AdaBN | 6 | 0.458 | 0.511 | 0.053 | 0.097 | 3/6 |
| ColonDB | Source-only continuation | 6 | 0.514 | 0.544 | 0.030 | 0.053 | 2/6 |
| ColonDB | Control | 6 | 0.435 | 0.504 | 0.069 | 0.133 | 3/6 |
| ColonDB | SourceOnly | 6 | 0.444 | 0.502 | 0.058 | 0.118 | 3/6 |
| PolypGen | DACS | 18 | 0.617 | 0.627 | 0.009 | 0.025 | 0/18 |
| PolypGen | DACS without paste | 6 | 0.624 | 0.634 | 0.010 | 0.019 | 0/6 |
| PolypGen | MinEnt+MMD | 18 | 0.615 | 0.620 | 0.005 | 0.017 | 0/18 |
| PolypGen | FDA | 18 | 0.629 | 0.633 | 0.004 | 0.018 | 0/18 |
| PolypGen | AdaBN | 6 | 0.504 | 0.534 | 0.030 | 0.058 | 1/6 |
| PolypGen | Source-only continuation | 6 | 0.601 | 0.620 | 0.019 | 0.028 | 0/6 |
| PolypGen | Control | 6 | 0.615 | 0.620 | 0.005 | 0.027 | 0/6 |
| PolypGen | SourceOnly | 6 | 0.582 | 0.599 | 0.016 | 0.042 | 0/6 |
| FAR | DACS | 6 | 0.565 | 0.574 | 0.009 | 0.024 | 0/6 |
| FAR | DACS without paste | 6 | 0.570 | 0.579 | 0.009 | 0.024 | 0/6 |
| FAR | DACS without confidence weight | 6 | 0.570 | 0.576 | 0.007 | 0.028 | 0/6 |
| FAR | MinEnt+MMD | 6 | 0.542 | 0.553 | 0.011 | 0.028 | 0/6 |
| FAR | FDA | 6 | 0.563 | 0.568 | 0.005 | 0.014 | 0/6 |
| FAR | AdaBN | 6 | 0.426 | 0.435 | 0.010 | 0.021 | 0/6 |
| FAR | Source-only continuation | 6 | 0.533 | 0.551 | 0.018 | 0.047 | 0/6 |
| FAR | Control | 6 | 0.492 | 0.513 | 0.021 | 0.113 | 1/6 |
| FAR | SourceOnly | 6 | 0.510 | 0.519 | 0.009 | 0.026 | 0/6 |

## 2. Selection across methods (not used for deployment)

Each pool holds every saved checkpoint of the listed methods on that target (all lambdas and base runs). **Original protocol pool** = the 13 pools of the original protocol (SourceOnly, 11 methods, the continued-training control) restricted to those run on the target. **Complete pool** = the original pool plus the later source-only continuation and DACS ablations (no paste; no confidence weight on ClinicDB and FAR). Oracle = best candidate by target labels. Loss = Oracle Dice − picked Dice, 95% interval from the group bootstrap in analyze.py. Rank = Source-Risk's position among the validators scored for every candidate (1 = smallest loss). `clinicdb/clinicdb_full.md` is the v1 analysis of ClinicDB's 1,860-candidate pool, the only adapted pool with all 19 validators (ClinicDB has no empty masks, so v1 = v2 there). The loss interval resamples the Dice groups: ClinicDB: 20 videos; ETIS: 196 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow); ColonDB: 380 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow); PolypGen: 6 centres; FAR: 186 patients with polyp images.

| Target | Pool | Methods | Candidates | Validators scored | Oracle Dice | Source-Risk pick | Source-Risk loss [95% CI] | Source-Risk rank | Best other validator | Worst validator |
|---|---|---:|---:|---|---|---|---|---:|---|---|
| ClinicDB | SourceOnly only | 1 | 180 | 19/19 | 0.684 (source_only) | source_only 0.684 | 0.000 [0.000, 0.000] | 1/19 | cmr_nhd_fg_ttd: source_only 0.684 (loss 0.000) | mixval: source_only 0.115 (loss 0.570) |
| ClinicDB | original protocol pool | 13 | 2220 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.727 (dacs) | continue 0.676 | 0.051 [-0.020, 0.099] | 1/9 | cmr_nhd_fg_gauss: dacs 0.670 (loss 0.057) | cmr_nhd_gauss: source_only 0.146 (loss 0.581) |
| ClinicDB | complete pool (+ later controls and ablations) | 16 | 2400 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.732 (dacs_nopaste) | continue 0.676 | 0.055 [0.001, 0.090] | 1/9 | cmr_nhd_fg_gauss: dacs 0.670 (loss 0.062) | cmr_nhd_gauss: source_only 0.146 (loss 0.586) |
| ETIS | SourceOnly only | 1 | 180 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.423 (source_only) | source_only 0.374 | 0.050 [0.016, 0.086] | 1/9 | entropy: source_only 0.314 (loss 0.110) | cmr_ei_gauss: source_only 0.137 (loss 0.286) |
| ETIS | original protocol pool | 9 | 1500 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.479 (minent) | dacs 0.410 | 0.068 [0.028, 0.109] | 1/9 | entropy: minent 0.365 (loss 0.114) | cmr_nhd_gauss: source_only 0.137 (loss 0.341) |
| ETIS | complete pool (+ later controls and ablations) | 11 | 1620 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.479 (minent) | dacs 0.410 | 0.068 [0.028, 0.106] | 1/9 | entropy: minent 0.365 (loss 0.114) | cmr_nhd_gauss: source_only 0.137 (loss 0.341) |
| ColonDB | SourceOnly only | 1 | 180 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.534 (source_only) | source_only 0.500 | 0.034 [0.011, 0.056] | 1/9 | entropy: source_only 0.383 (loss 0.151) | cmr_nhd_gauss: source_only 0.043 (loss 0.492) |
| ColonDB | original protocol pool | 9 | 1500 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.711 (dacs) | continue 0.488 | 0.224 [0.195, 0.254] | 1/9 | mcc: minent 0.386 (loss 0.325) | cmr_nhd_gauss: source_only 0.043 (loss 0.668) |
| ColonDB | complete pool (+ later controls and ablations) | 11 | 1620 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.711 (dacs) | continue 0.488 | 0.224 [0.196, 0.253] | 1/9 | mcc: minent 0.386 (loss 0.325) | cmr_nhd_gauss: source_only 0.043 (loss 0.668) |
| PolypGen | SourceOnly only | 1 | 180 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.616 (source_only) | source_only 0.616 | 0.000 [0.000, 0.000] | 1/9 | infomax: source_only 0.546 (loss 0.070) | cmr_nhd_gauss: source_only 0.403 (loss 0.213) |
| PolypGen | original protocol pool | 6 | 660 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.656 (fda) | dacs 0.632 | 0.025 [0.020, 0.036] | 1/9 | mcc: minent_mmd 0.599 (loss 0.057) | cmr_nhd_gauss: source_only 0.403 (loss 0.254) |
| PolypGen | complete pool (+ later controls and ablations) | 8 | 750 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.656 (fda) | dacs 0.632 | 0.025 [0.020, 0.037] | 1/9 | mcc: minent_mmd 0.599 (loss 0.057) | cmr_nhd_gauss: source_only 0.403 (loss 0.254) |
| FAR | SourceOnly only | 1 | 180 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.533 (source_only) | source_only 0.521 | 0.012 [-0.027, 0.041] | 1/9 | entropy: source_only 0.491 (loss 0.043) | cmr_ei_fg_gauss: source_only 0.245 (loss 0.289) |
| FAR | original protocol pool | 6 | 480 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.592 (fda) | continue 0.539 | 0.053 [0.002, 0.060] | 3/9 | cmr_ei_gauss: dacs 0.564 (loss 0.028) | cmr_ei_fg_gauss: source_only 0.245 (loss 0.347) |
| FAR | complete pool (+ later controls and ablations) | 9 | 600 | 9/19 (missing: iwcv, dev, devn, snd, mixval, mixval_fg, cmr_nhd_ttd, cmr_ei_ttd, cmr_nhd_fg_ttd, cmr_ei_fg_ttd) | 0.592 (fda) | dacs_nopaste 0.588 | 0.004 [-0.043, 0.020] | 1/9 | entropy: dacs_nopaste 0.562 (loss 0.030) | cmr_ei_fg_gauss: source_only 0.245 (loss 0.347) |
