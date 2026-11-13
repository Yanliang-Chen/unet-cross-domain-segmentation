# Deployment rule (amendment v2, post hoc)

Rule: (method, λ) with the highest mean deployed Dice on ClinicDB among {DACS, MinEnt+MMD, FDA} × {0.1, 0.5, 1}; checkpoint = Source-Risk pick per trajectory; single model = the base run whose pick has the highest Source-Risk. Every target had been seen when the rule was written, so this is not an independent confirmation. Dice, Small Dice and FPR columns are image means; the gain column is paired and bootstrapped over base runs and groups (ClinicDB: 20 videos; ETIS: 196 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow); ColonDB: 380 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow); PolypGen: 6 centres; FAR: 186 patients with polyp images).

ClinicDB mean deployed Dice by (method, λ): DACS λ1 0.699; DACS λ0.5 0.690; MinEnt+MMD λ0.5 0.674; MinEnt+MMD λ1 0.671; DACS λ0.1 0.669; MinEnt+MMD λ0.1 0.666; FDA λ0.1 0.663; FDA λ1 0.650; FDA λ0.5 0.637

**Rule picks DACS, λ=1.**

| Target | λ used | Single model (run, step) | Dice | Small Dice | Normal FPR | Empty FPR | 6-run mean Dice [min, max] | Gain vs SourceOnly, 6 runs [95% CI] | SourceOnly single model Dice / FPR |
|---|---:|---|---:|---:|---:|---:|---|---|---|
| ClinicDB | 1 | a1_seed42 step 600 | 0.687 | 0.477 | – | – | 0.699 [0.687, 0.707] | +0.093 [+0.045, +0.151] (6/6) | 0.684 / – |
| ETIS | 1 | a1_seed42 step 1600 | 0.468 | 0.365 | – | – | 0.422 [0.377, 0.468] | +0.056 [+0.025, +0.087] (6/6) | 0.374 / – |
| ColonDB | 1 | a1_seed42 step 2000 | 0.683 | 0.489 | – | – | 0.658 [0.611, 0.705] | +0.214 [+0.187, +0.245] (6/6) | 0.500 / – |
| PolypGen | 1 | a1_seed42 step 1600 | 0.631 | 0.410 | 16.0% | 43.2% | 0.625 [0.603, 0.637] | +0.040 [+0.022, +0.061] (6/6) | 0.616 / 30.8% |
| FAR | 0.5 (only λ run) | a1_seed42 step 1600 | 0.570 | 0.368 | 45.6% | – | 0.565 [0.555, 0.574] | +0.063 [+0.035, +0.091] (6/6) | 0.521 / 59.7% |
