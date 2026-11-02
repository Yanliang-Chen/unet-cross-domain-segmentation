# Paired gains on far: Source-Risk pick per trajectory

Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---:|---|---:|---|---:|
| dacs | 6 | 0.565 | 0.450 | +0.079 [+0.045, +0.124] | 6/6 | +0.063 [+0.036, +0.092] | 6/6 |
| fda | 6 | 0.563 | 0.544 | +0.066 [+0.024, +0.116] | 6/6 | +0.050 [+0.020, +0.084] | 6/6 |
| minent_mmd | 6 | 0.542 | 0.652 | +0.054 [+0.021, +0.095] | 6/6 | +0.038 [+0.011, +0.065] | 6/6 |
| source_only | 6 | 0.510 | 0.565 | +0.016 [-0.019, +0.051] | 4/6 | +0.000 [+0.000, +0.000] | 0/6 |
| continue | 6 | 0.492 | 0.737 | +0.000 [+0.000, +0.000] | 0/6 | -0.016 [-0.051, +0.017] | 2/6 |
| adabn | 6 | 0.426 | 0.906 | -0.065 [-0.097, -0.028] | 0/6 | -0.081 [-0.114, -0.048] | 0/6 |
