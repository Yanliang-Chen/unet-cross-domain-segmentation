# Paired gains on polypgen: Source-Risk pick per trajectory (pre-specified lambda 0.5 only)

Gain = per-group Dice difference to the same base run's pick, lambda 0.5 only (pre-specified reduced grid); 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---:|---|---:|---|---:|
| fda | 6 | 0.628 | 0.364 | +0.019 [-0.003, +0.044] | 4/6 | +0.046 [+0.021, +0.071] | 6/6 |
| dacs | 6 | 0.618 | 0.255 | +0.005 [-0.016, +0.027] | 4/6 | +0.032 [+0.010, +0.054] | 6/6 |
| continue | 6 | 0.615 | 0.791 | +0.000 [+0.000, +0.000] | 0/6 | +0.027 [+0.009, +0.044] | 6/6 |
| minent_mmd | 6 | 0.614 | 0.340 | +0.000 [-0.020, +0.023] | 2/6 | +0.027 [+0.008, +0.047] | 6/6 |
| source_only | 6 | 0.582 | 0.247 | -0.027 [-0.044, -0.008] | 0/6 | +0.000 [+0.000, +0.000] | 0/6 |
| adabn | 6 | 0.504 | 0.822 | -0.086 [-0.141, -0.028] | 0/6 | -0.059 [-0.117, -0.003] | 0/6 |
