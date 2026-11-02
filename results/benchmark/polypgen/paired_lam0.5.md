# Paired gains on polypgen: Source-Risk pick per trajectory (pre-specified lambda 0.5 only)

Gain = per-group Dice difference to the same base run's pick, averaged over lambdas; 95% interval from 2000 joint bootstrap draws over base runs and image groups.

| Algorithm | Trajectories | Pick Dice | Normal FPR | vs control [95% CI] | Wins | vs SourceOnly [95% CI] | Wins |
|---|---:|---:|---:|---|---:|---|---:|
| fda | 6 | 0.611 | 0.364 | +0.025 [-0.002, +0.056] | 5/6 | +0.037 [+0.016, +0.059] | 6/6 |
| minent_mmd | 6 | 0.603 | 0.340 | +0.015 [-0.011, +0.048] | 4/6 | +0.027 [+0.011, +0.044] | 6/6 |
| dacs | 6 | 0.601 | 0.255 | +0.013 [-0.008, +0.034] | 4/6 | +0.025 [-0.004, +0.051] | 6/6 |
| continue | 6 | 0.592 | 0.791 | +0.000 [+0.000, +0.000] | 0/6 | +0.012 [-0.021, +0.037] | 5/6 |
| source_only | 6 | 0.573 | 0.247 | -0.012 [-0.038, +0.026] | 1/6 | +0.000 [+0.000, +0.000] | 0/6 |
| adabn | 6 | 0.469 | 0.822 | -0.106 [-0.157, -0.054] | 0/6 | -0.094 [-0.142, -0.037] | 0/6 |
