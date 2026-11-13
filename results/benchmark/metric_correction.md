# Metric correction (protocol amendment v2, 2026-11-13)

v1 counted PolypGen's 125 polyp-set images with an **empty ground-truth mask** (of 1,537) as polyp
images, and, because their area 0 is below 2%, as small polyps (125 of 418). On those images Dice
is 1 for an empty prediction and 0 otherwise, so v1 mixed "suppress predictions on polyp-free
frames" into segmentation Dice. v2 counts only images with a non-empty mask (1,412; small polyps
293) and reports the empty-mask images separately as a false-positive rate. ClinicDB, ETIS,
ColonDB and FAR have no empty masks; their numbers are unchanged.

## PolypGen, deployed Source-Risk picks

| Method | Dice v1 → v2 | vs control v1 → v2 [95% CI] | vs SourceOnly v1 → v2 [95% CI] | Small Dice v1 → v2 | Empty-mask FPR (v2 only) |
|---|---|---|---|---|---:|
| FDA | 0.611 → 0.629 | +0.025 → +0.021 [−0.005, +0.048] | +0.037 → +0.048 [+0.021, +0.074] | 0.366 → 0.346 | 58.8% |
| DACS | 0.599 → 0.617 | +0.011 → +0.005 [−0.014, +0.024] | +0.023 → +0.032 [+0.014, +0.052] | 0.364 → 0.351 | 60.7% |
| MinEnt+MMD | 0.603 → 0.615 | +0.014 → +0.002 [−0.016, +0.020] | +0.026 → +0.029 [+0.014, +0.045] | 0.381 → 0.345 | 53.4% |
| Control | 0.592 → 0.615 | 0 | +0.012 → +0.027 [+0.009, +0.044] | 0.341 → 0.343 | 66.3% |
| SourceOnly | 0.573 → 0.582 | −0.012 → −0.027 [−0.044, −0.008] | 0 | 0.358 → 0.310 | 53.1% |

## Conclusions that change

1. **DACS on PolypGen now beats SourceOnly** (interval above zero), so all three methods beat
   SourceOnly on all five targets. The control also beats SourceOnly on PolypGen.
2. **No method beats the control on PolypGen**; the v1 lead over the control was partly the
   control's extra false positives on empty-mask frames (66% vs 53–61%).
3. **Post-processing does not raise PolypGen Dice.** v1: +0.005 to +0.015, attributed to small
   polyps. v2: −0.001 to +0.000; small-polyp Dice is unchanged (DACS 0.351 → 0.351). The v1 gain
   came from empty-mask frames, where deleting a blob turns Dice 0 into 1.
4. **Small polyps on PolypGen.** v1: gains near zero and the control lower than SourceOnly. v2:
   every method and the control are 0.03–0.04 above SourceOnly (DACS +0.041, 16/18; control
   +0.033, 5/6). "Continued training hurts small polyps everywhere" no longer holds on PolypGen.
5. **Lambda on PolypGen.** FDA is best at λ=0.1 (0.638 / 0.628 / 0.622) and DACS improves with λ
   (0.610 / 0.618 / 0.625); MinEnt+MMD is flat. Directions as in v1; levels shift up.

Normal-image FPR (PolypGen 1,755 normals) and every number on the other four targets are unchanged.
v1 tables are in git history (commit e0e7903 and earlier).
