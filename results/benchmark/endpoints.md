# Endpoints vs SourceOnly (amendment v2)

Deployed Source-Risk picks; each cell is the method's mean level, then the paired difference to the same base run's SourceOnly pick with its 95% interval (joint bootstrap over base runs and groups). Dice column uses report.py's intervals; the other columns use a fixed seed per comparison. FPR in percentage points. Verdict per amendment_v2.endpoints: Dice interval above zero = Dice better; normal-FPR (any pixel) interval below zero = FPR lower. Levels are image means; differences are group-weighted (video/centre/patient where available), so a difference can disagree with the gap between two levels when groups differ in size (PolypGen empty-mask images come from few groups).

## ClinicDB (427 polyp images, 36 small)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups resampled per endpoint: Dice: 20 videos; small-polyp Dice: 9 videos.

| Method | Dice | Small-polyp Dice | Verdict |
|---|---|---|---|
| SourceOnly | 0.610 | 0.449 | reference |
| DACS | 0.686 +0.082 [+0.034, +0.138] | 0.458 +0.011 [-0.107, +0.117] | Dice better |
| MinEnt+MMD | 0.670 +0.068 [+0.020, +0.126] | 0.453 -0.001 [-0.094, +0.082] | Dice better |
| FDA | 0.650 +0.042 [-0.007, +0.094] | 0.470 +0.022 [-0.101, +0.136] | not enough evidence |
| MinEnt | 0.664 +0.058 [+0.017, +0.109] | 0.478 +0.038 [-0.050, +0.132] | Dice better |
| MMD | 0.658 +0.056 [+0.004, +0.116] | 0.412 -0.053 [-0.147, +0.034] | Dice better |
| DANN | 0.649 +0.046 [-0.006, +0.108] | 0.416 -0.049 [-0.158, +0.052] | not enough evidence |
| Control | 0.637 +0.031 [-0.027, +0.096] | 0.393 -0.077 [-0.190, +0.031] | not enough evidence |
| Source-only continuation | 0.654 +0.045 [+0.003, +0.093] | 0.433 -0.037 [-0.135, +0.055] | Dice better |
| AdaBN | 0.606 +0.008 [-0.080, +0.096] | 0.270 -0.134 [-0.294, +0.024] | not enough evidence |

## ETIS (196 polyp images, 100 small)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups resampled per endpoint: Dice: 196 single images; small-polyp Dice: 100 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow).

| Method | Dice | Small-polyp Dice | Verdict |
|---|---|---|---|
| SourceOnly | 0.366 | 0.222 | reference |
| DACS | 0.403 +0.038 [+0.014, +0.065] | 0.258 +0.036 [-0.001, +0.069] | Dice better |
| MinEnt+MMD | 0.412 +0.046 [+0.017, +0.080] | 0.259 +0.037 [-0.003, +0.082] | Dice better |
| FDA | 0.430 +0.064 [+0.033, +0.100] | 0.299 +0.077 [+0.033, +0.122] | Dice better |
| MinEnt | 0.393 +0.027 [+0.000, +0.055] | 0.237 +0.015 [-0.018, +0.051] | Dice better |
| MMD | 0.388 +0.022 [-0.007, +0.055] | 0.220 -0.002 [-0.045, +0.041] | not enough evidence |
| DANN | 0.362 -0.004 [-0.037, +0.028] | 0.187 -0.035 [-0.076, +0.004] | not enough evidence |
| Control | 0.338 -0.028 [-0.071, +0.011] | 0.159 -0.063 [-0.116, -0.010] | not enough evidence |
| Source-only continuation | 0.379 +0.014 [-0.013, +0.041] | 0.244 +0.022 [-0.013, +0.056] | not enough evidence |
| AdaBN | 0.334 -0.032 [-0.076, +0.018] | 0.171 -0.051 [-0.106, +0.002] | not enough evidence |

## ColonDB (380 polyp images, 97 small)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups resampled per endpoint: Dice: 380 single images; small-polyp Dice: 97 single images (no video ids, so adjacent frames count as independent and intervals may be too narrow).

| Method | Dice | Small-polyp Dice | Verdict |
|---|---|---|---|
| SourceOnly | 0.444 | 0.258 | reference |
| DACS | 0.609 +0.165 [+0.135, +0.197] | 0.373 +0.116 [+0.055, +0.181] | Dice better |
| MinEnt+MMD | 0.568 +0.124 [+0.086, +0.168] | 0.341 +0.083 [+0.025, +0.148] | Dice better |
| FDA | 0.508 +0.064 [+0.029, +0.101] | 0.345 +0.087 [+0.030, +0.152] | Dice better |
| MinEnt | 0.534 +0.089 [+0.049, +0.130] | 0.345 +0.088 [+0.037, +0.146] | Dice better |
| MMD | 0.541 +0.097 [+0.058, +0.141] | 0.299 +0.042 [-0.018, +0.109] | Dice better |
| DANN | 0.498 +0.053 [+0.001, +0.101] | 0.294 +0.037 [-0.046, +0.118] | Dice better |
| Control | 0.435 -0.009 [-0.068, +0.040] | 0.239 -0.019 [-0.101, +0.062] | not enough evidence |
| Source-only continuation | 0.514 +0.070 [+0.033, +0.108] | 0.335 +0.078 [+0.023, +0.145] | Dice better |
| AdaBN | 0.458 +0.014 [-0.031, +0.058] | 0.227 -0.030 [-0.102, +0.039] | not enough evidence |

## PolypGen (1412 polyp images, 293 small, 1755 normal, 125 empty-mask)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups resampled per endpoint: Dice: 6 centres; small-polyp Dice: 6 centres; normal-image FPR: 10 negative sequences; empty-mask FPR: 6 centres.

| Method | Dice | Small-polyp Dice | Normal FPR (any) | Normal FPR (area ≥0.1%) | Empty-mask FPR | Verdict |
|---|---|---|---|---|---|---|
| SourceOnly | 0.582 | 0.310 | 24.7% | 21.6% | 53.1% | reference |
| DACS | 0.617 +0.032 [+0.014, +0.052] | 0.351 +0.041 [+0.010, +0.072] | 36.8% +11.2 [+3.4, +20.9] | 32.5% +10.3 [+2.4, +19.7] | 60.7% -3.2 [-19.4, +9.7] | trade-off (Dice ↑, FPR ↑) |
| MinEnt+MMD | 0.615 +0.029 [+0.014, +0.045] | 0.345 +0.032 [+0.003, +0.062] | 39.7% +15.1 [+3.0, +29.8] | 35.4% +13.8 [+1.5, +28.5] | 53.4% -3.7 [-15.6, +6.2] | trade-off (Dice ↑, FPR ↑) |
| FDA | 0.629 +0.048 [+0.021, +0.074] | 0.346 +0.044 [+0.018, +0.069] | 37.1% +10.7 [+0.8, +19.2] | 32.6% +9.3 [-0.3, +17.9] | 58.8% +6.3 [-5.3, +18.3] | trade-off (Dice ↑, FPR ↑) |
| Control | 0.615 +0.027 [+0.009, +0.044] | 0.343 +0.028 [-0.015, +0.061] | 79.1% +52.7 [+39.7, +65.0] | 74.4% +50.8 [+36.9, +64.5] | 66.3% +0.4 [-18.5, +15.3] | trade-off (Dice ↑, FPR ↑) |
| Source-only continuation | 0.601 +0.016 [-0.001, +0.033] | 0.338 +0.025 [-0.010, +0.057] | 26.9% +2.1 [-4.2, +8.4] | 23.4% +1.8 [-3.7, +7.7] | 45.5% -11.6 [-26.1, -0.4] | not enough evidence |
| AdaBN | 0.504 -0.059 [-0.117, -0.003] | 0.196 -0.110 [-0.171, -0.051] | 82.2% +57.9 [+49.3, +66.8] | 80.2% +59.0 [+50.1, +68.7] | 92.4% +36.1 [+14.4, +58.6] | worse (Dice ↓, FPR ↑) |

## FAR (432 polyp images, 123 small, 7749 normal)

Levels are image means. Paired differences are averaged per group and bootstrapped jointly over base runs and groups; groups resampled per endpoint: Dice: 186 patients with polyp images; small-polyp Dice: 93 patients; normal-image FPR: 444 patients.

| Method | Dice | Small-polyp Dice | Normal FPR (any) | Normal FPR (area ≥0.1%) | Verdict |
|---|---|---|---|---|---|
| SourceOnly | 0.510 | 0.292 | 56.5% | 51.8% | reference |
| DACS | 0.565 +0.063 [+0.036, +0.092] | 0.356 +0.066 [+0.025, +0.110] | 45.0% -10.8 [-17.8, -2.6] | 38.7% -12.2 [-18.5, -4.3] | both |
| MinEnt+MMD | 0.542 +0.038 [+0.011, +0.065] | 0.323 +0.030 [-0.015, +0.079] | 65.2% +8.5 [+0.7, +15.8] | 59.4% +7.6 [-0.9, +15.3] | trade-off (Dice ↑, FPR ↑) |
| FDA | 0.563 +0.050 [+0.020, +0.084] | 0.337 +0.043 [-0.006, +0.095] | 54.4% -1.3 [-8.9, +5.9] | 48.8% -2.0 [-9.1, +4.3] | Dice better |
| Control | 0.492 -0.016 [-0.051, +0.017] | 0.256 -0.032 [-0.084, +0.023] | 73.7% +16.6 [+13.0, +21.1] | 69.1% +16.8 [+12.5, +21.5] | FPR higher |
| Source-only continuation | 0.533 +0.031 [+0.003, +0.058] | 0.316 +0.031 [-0.023, +0.091] | 54.3% -1.9 [-6.5, +3.4] | 48.8% -2.6 [-6.9, +2.5] | Dice better |
| AdaBN | 0.426 -0.081 [-0.114, -0.048] | 0.190 -0.091 [-0.142, -0.042] | 90.6% +33.3 [+28.9, +37.5] | 88.7% +36.3 [+31.8, +40.2] | worse (Dice ↓, FPR ↑) |
