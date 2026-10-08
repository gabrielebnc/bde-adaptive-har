# Pareto evaluation: validation

FLOPs = 2 * Conv1d/Linear MACs, counted from executed tensor shapes. Excludes bias, BatchNorm, ReLU, residual addition, pooling, softmax and routing. Static paths skip unused heads; adaptive paths pay for every visited head. These are consistently calculated operation counts, not latency or energy.

Each dot is an evaluated configuration. The frontier is empirical over these points; lines do not imply measured intermediate policies or statistical significance.

| Configuration | Accuracy (%) | Average FLOPs/window | Non-dominated |
| --- | ---: | ---: | --- |
| normal_0.0_0.0 | 91.020 | 81456.0 | True |
| normal_0.0_0.5 | 91.020 | 81456.0 | True |
| normal_0.0_0.6 | 91.020 | 81456.0 | True |
| normal_0.0_0.7 | 91.020 | 81456.0 | True |
| normal_0.0_0.8 | 91.020 | 81456.0 | True |
| normal_0.0_0.9 | 91.020 | 81456.0 | True |
| normal_0.0_0.95 | 91.020 | 81456.0 | True |
| normal_0.0_0.98 | 91.020 | 81456.0 | True |
| normal_0.0_0.99 | 91.020 | 81456.0 | True |
| normal_0.0_1.0 | 91.020 | 81456.0 | True |
| normal_0.5_0.0 | 91.276 | 81677.6 | True |
| normal_0.5_0.5 | 91.327 | 81688.0 | True |
| normal_0.5_0.6 | 91.327 | 81688.0 | True |
| normal_0.5_0.7 | 91.327 | 81688.0 | True |
| normal_0.5_0.8 | 91.327 | 81698.5 | False |
| normal_0.5_0.9 | 91.327 | 81698.5 | False |
| normal_0.5_0.95 | 91.327 | 81698.5 | False |
| normal_0.5_0.98 | 91.327 | 81698.5 | False |
| normal_0.5_0.99 | 91.327 | 81708.9 | False |
| normal_0.5_1.0 | 91.327 | 81761.1 | False |
| normal_0.6_0.0 | 91.531 | 85749.5 | True |
| normal_0.6_0.5 | 91.582 | 85759.9 | True |
| normal_0.6_0.6 | 92.143 | 86292.5 | True |
| normal_0.6_0.7 | 92.143 | 86804.2 | False |
| normal_0.6_0.8 | 92.143 | 87044.4 | False |
| normal_0.6_0.9 | 92.092 | 87075.7 | False |
| normal_0.6_0.95 | 92.041 | 87086.2 | False |
| normal_0.6_0.98 | 92.041 | 87086.2 | False |
| normal_0.6_0.99 | 92.041 | 87096.6 | False |
| normal_0.6_1.0 | 92.041 | 87368.1 | False |
| normal_0.7_0.0 | 92.449 | 90153.8 | True |
| normal_0.7_0.5 | 92.500 | 90174.7 | True |
| normal_0.7_0.6 | 94.031 | 91480.0 | True |
| normal_0.7_0.7 | 94.031 | 92044.0 | False |
| normal_0.7_0.8 | 94.082 | 92451.2 | True |
| normal_0.7_0.9 | 94.031 | 92858.5 | False |
| normal_0.7_0.95 | 93.929 | 93025.6 | False |
| normal_0.7_0.98 | 93.929 | 93036.0 | False |
| normal_0.7_0.99 | 93.929 | 93067.4 | False |
| normal_0.7_1.0 | 93.929 | 93432.9 | False |
| normal_0.8_0.0 | 92.755 | 95444.5 | False |
| normal_0.8_0.5 | 92.806 | 95465.4 | False |
| normal_0.8_0.6 | 94.337 | 96937.8 | True |
| normal_0.8_0.7 | 94.388 | 97647.9 | True |
| normal_0.8_0.8 | 94.439 | 98452.0 | True |
| normal_0.8_0.9 | 94.388 | 99600.8 | False |
| normal_0.8_0.95 | 94.286 | 100018.5 | False |
| normal_0.8_0.98 | 94.286 | 100143.8 | False |
| normal_0.8_0.99 | 94.235 | 100185.6 | False |
| normal_0.8_1.0 | 94.133 | 100718.1 | False |
| normal_0.9_0.0 | 93.061 | 99267.1 | False |
| normal_0.9_0.5 | 93.214 | 99308.9 | False |
| normal_0.9_0.6 | 94.796 | 100791.8 | True |
| normal_0.9_0.7 | 94.847 | 101501.9 | True |
| normal_0.9_0.8 | 94.898 | 102337.3 | True |
| normal_0.9_0.9 | 94.847 | 103611.3 | False |
| normal_0.9_0.95 | 94.745 | 104561.6 | False |
| normal_0.9_0.98 | 94.643 | 104937.6 | False |
| normal_0.9_0.99 | 94.592 | 105083.8 | False |
| normal_0.9_1.0 | 94.490 | 105981.9 | False |
| normal_0.95_0.0 | 93.163 | 101510.8 | False |
| normal_0.95_0.5 | 93.316 | 101552.6 | False |
| normal_0.95_0.6 | 94.898 | 103035.5 | False |
| normal_0.95_0.7 | 94.949 | 103745.6 | True |
| normal_0.95_0.8 | 95.000 | 104601.9 | True |
| normal_0.95_0.9 | 95.000 | 105896.8 | False |
| normal_0.95_0.95 | 94.949 | 106941.1 | False |
| normal_0.95_0.98 | 94.796 | 107411.0 | False |
| normal_0.95_0.99 | 94.745 | 107619.9 | False |
| normal_0.95_1.0 | 94.643 | 109061.0 | False |
| normal_0.98_0.0 | 93.214 | 105804.3 | False |
| normal_0.98_0.5 | 93.367 | 105846.1 | False |
| normal_0.98_0.6 | 94.949 | 107339.4 | False |
| normal_0.98_0.7 | 95.000 | 108049.5 | False |
| normal_0.98_0.8 | 95.051 | 108926.7 | True |
| normal_0.98_0.9 | 95.051 | 110232.1 | False |
| normal_0.98_0.95 | 94.949 | 111286.8 | False |
| normal_0.98_0.98 | 94.796 | 112268.4 | False |
| normal_0.98_0.99 | 94.745 | 112581.7 | False |
| normal_0.98_1.0 | 94.643 | 114941.8 | False |
| normal_0.99_0.0 | 93.214 | 111150.4 | False |
| normal_0.99_0.5 | 93.367 | 111192.2 | False |
| normal_0.99_0.6 | 94.949 | 112685.5 | False |
| normal_0.99_0.7 | 95.000 | 113395.6 | False |
| normal_0.99_0.8 | 95.051 | 114272.8 | False |
| normal_0.99_0.9 | 95.051 | 115578.2 | False |
| normal_0.99_0.95 | 94.949 | 116632.9 | False |
| normal_0.99_0.98 | 94.796 | 117719.0 | False |
| normal_0.99_0.99 | 94.745 | 118063.6 | False |
| normal_0.99_1.0 | 94.643 | 122219.8 | False |
| normal_1.0_0.0 | 93.214 | 135748.0 | False |
| normal_1.0_0.5 | 93.367 | 135789.8 | False |
| normal_1.0_0.6 | 94.949 | 137283.1 | False |
| normal_1.0_0.7 | 95.000 | 137993.2 | False |
| normal_1.0_0.8 | 95.051 | 138870.4 | False |
| normal_1.0_0.9 | 95.051 | 140175.8 | False |
| normal_1.0_0.95 | 94.949 | 141240.9 | False |
| normal_1.0_0.98 | 94.796 | 142327.0 | False |
| normal_1.0_0.99 | 94.745 | 142682.1 | False |
| normal_1.0_1.0 | 94.643 | 150660.4 | False |
| low-power_0.0_off | 91.020 | 81456.0 | True |
| low-power_0.5_off | 91.276 | 81677.6 | True |
| low-power_0.6_off | 91.531 | 85749.5 | True |
| low-power_0.7_off | 92.449 | 90153.8 | True |
| low-power_0.8_off | 92.755 | 95444.5 | False |
| low-power_0.9_off | 93.061 | 99267.1 | False |
| low-power_0.95_off | 93.163 | 101510.8 | False |
| low-power_0.98_off | 93.214 | 105804.3 | False |
| low-power_0.99_off | 93.214 | 111150.4 | False |
| low-power_1.0_off | 93.214 | 135748.0 | False |
| static_exit_1 | 91.020 | 81456.0 | True |
| static_exit_2 | 93.214 | 135700.0 | False |
| static_exit_3 | 94.643 | 156084.0 | False |

## Interpretation and limits

The evaluated family spans 81,456 to 156,084 FLOPs/window. 78 of 113 configurations are dominated in this sample (equal-coordinate ties are retained).

See baseline_comparison.json for accuracy differences (percentage points) and compute reductions relative to the fixed full-depth path. No policy is chosen using these test differences.

The three static references use the frozen baseline checkpoint and skip unused heads. They are fixed-depth ablations, not separately trained or tuned static CNNs. A stronger independently trained static comparison and a compression experiment remain project work.

The included checkpoint is exploratory and single-seed; its documentation discloses prior inspection of test subjects. This workflow prevents new test-threshold tuning but cannot make those subjects a fresh holdout. Small differences need repeated runs/uncertainty analysis.

Synthetic or smoke-run results are only software checks, not evidence of HAR performance. Changing training, compression, preprocessing or calibration requires a new validation manifest.

## Descriptive matched-accuracy comparison

Among the evaluated adaptive configurations, `normal_0.9_0.98` is closest in accuracy to the full-depth static reference: +0.000 percentage points, with 32.77% fewer calculated FLOPs. This is a description of the plotted results, not a test-selected deployment policy or evidence of statistical equivalence. Negative compute reduction means more work.
