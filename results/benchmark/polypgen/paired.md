# Paired gains on polypgen: Source-Risk pick per trajectory

Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---:|---|---:|---|---:|
| fda | 18 | 0.611 | 0.371 | +0.025 [-0.003, +0.058] | 14/18 | +0.037 [+0.015, +0.061] | 17/18 |
| minent_mmd | 18 | 0.603 | 0.397 | +0.014 [-0.010, +0.046] | 12/18 | +0.026 [+0.011, +0.043] | 18/18 |
| dacs | 18 | 0.599 | 0.368 | +0.011 [-0.007, +0.030] | 12/18 | +0.023 [-0.009, +0.048] | 17/18 |
| continue | 6 | 0.592 | 0.791 | +0.000 [+0.000, +0.000] | 0/6 | +0.012 [-0.021, +0.037] | 5/6 |
| source_only | 6 | 0.573 | 0.247 | -0.012 [-0.038, +0.026] | 1/6 | +0.000 [+0.000, +0.000] | 0/6 |
| adabn | 6 | 0.469 | 0.822 | -0.106 [-0.157, -0.054] | 0/6 | -0.094 [-0.142, -0.037] | 0/6 |
