# Paired gains on polypgen: Source-Risk pick per trajectory

Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---:|---|---:|---|---:|
| fda | 18 | 0.629 | 0.371 | +0.021 [-0.005, +0.048] | 13/18 | +0.048 [+0.021, +0.074] | 18/18 |
| dacs | 18 | 0.617 | 0.368 | +0.005 [-0.014, +0.024] | 12/18 | +0.032 [+0.014, +0.052] | 17/18 |
| minent_mmd | 18 | 0.615 | 0.397 | +0.002 [-0.016, +0.020] | 9/18 | +0.029 [+0.014, +0.045] | 18/18 |
| continue | 6 | 0.615 | 0.791 | +0.000 [+0.000, +0.000] | 0/6 | +0.027 [+0.009, +0.044] | 6/6 |
| continue_source | 6 | 0.601 | 0.269 | -0.010 [-0.028, +0.010] | 1/6 | +0.016 [-0.001, +0.033] | 4/6 |
| source_only | 6 | 0.582 | 0.247 | -0.027 [-0.044, -0.008] | 0/6 | +0.000 [+0.000, +0.000] | 0/6 |
| adabn | 6 | 0.504 | 0.822 | -0.086 [-0.141, -0.028] | 0/6 | -0.059 [-0.117, -0.003] | 0/6 |
