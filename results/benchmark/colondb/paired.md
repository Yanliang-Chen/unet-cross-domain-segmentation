# Paired gains on colondb: Source-Risk pick per trajectory

Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---|---:|---|---:|
| dacs | 18 | 0.609 | +0.174 [+0.130, +0.223] | 18/18 | +0.165 [+0.135, +0.197] | 18/18 |
| minent_mmd | 18 | 0.568 | +0.133 [+0.079, +0.192] | 18/18 | +0.124 [+0.086, +0.168] | 18/18 |
| mmd | 18 | 0.541 | +0.106 [+0.062, +0.158] | 18/18 | +0.097 [+0.058, +0.141] | 18/18 |
| minent | 18 | 0.534 | +0.098 [+0.038, +0.166] | 18/18 | +0.089 [+0.049, +0.130] | 17/18 |
| fda | 18 | 0.508 | +0.073 [+0.021, +0.131] | 15/18 | +0.064 [+0.029, +0.101] | 17/18 |
| dann | 18 | 0.498 | +0.062 [+0.029, +0.102] | 17/18 | +0.053 [+0.001, +0.101] | 16/18 |
| adabn | 6 | 0.458 | +0.023 [-0.013, +0.063] | 4/6 | +0.014 [-0.031, +0.058] | 4/6 |
| source_only | 6 | 0.444 | +0.009 [-0.044, +0.068] | 3/6 | +0.000 [+0.000, +0.000] | 0/6 |
| continue | 6 | 0.435 | +0.000 [+0.000, +0.000] | 0/6 | -0.009 [-0.068, +0.040] | 3/6 |
