# Pareto evaluation: test

FLOPs = 2 * Conv1d/Linear MACs, counted from executed tensor shapes. Excludes bias, BatchNorm, ReLU, residual addition, pooling, softmax and routing. Static paths skip unused heads; adaptive paths pay for every visited head. These are consistently calculated operation counts, not latency or energy.

Each dot is an evaluated configuration. The frontier is empirical over these points; lines do not imply measured intermediate policies or statistical significance.

| Configuration | Accuracy (%) | Average FLOPs/window | Non-dominated |
| --- | ---: | ---: | --- |
| normal_0.0_0.0 | 90.533 | 81456.0 | True |
| normal_0.5_0.0 | 90.601 | 81492.8 | True |
| normal_0.5_0.5 | 90.601 | 81492.8 | True |
| normal_0.6_0.0 | 90.261 | 85214.3 | False |
| normal_0.6_0.5 | 90.261 | 85214.3 | False |
| normal_0.6_0.6 | 90.329 | 85881.0 | False |
| normal_0.7_0.0 | 90.601 | 88069.8 | False |
| normal_0.7_0.5 | 90.567 | 88076.7 | False |
| normal_0.7_0.6 | 91.585 | 89215.8 | True |
| normal_0.7_0.8 | 91.585 | 89813.1 | False |
| normal_0.8_0.6 | 91.924 | 94388.6 | True |
| normal_0.8_0.7 | 91.924 | 95090.1 | False |
| normal_0.8_0.8 | 91.856 | 95638.8 | False |
| normal_0.9_0.6 | 92.229 | 99641.5 | True |
| normal_0.9_0.7 | 92.263 | 100370.8 | True |
| normal_0.9_0.8 | 92.128 | 100975.1 | False |
| normal_0.95_0.7 | 92.433 | 103723.8 | True |
| normal_0.95_0.8 | 92.297 | 104355.8 | False |
| normal_0.98_0.8 | 92.467 | 108462.0 | True |
| low-power_0.0_off | 90.533 | 81456.0 | True |
| low-power_0.5_off | 90.601 | 81492.8 | True |
| low-power_0.6_off | 90.261 | 85214.3 | False |
| low-power_0.7_off | 90.601 | 88069.8 | False |
| low-power_0.8_off | 90.872 | 93117.6 | False |
| low-power_0.9_off | 91.110 | 98349.7 | False |
| low-power_0.95_off | 91.279 | 101702.7 | False |
| low-power_0.98_off | 91.347 | 105774.1 | False |
| static_exit_1 | 90.533 | 81456.0 | True |
| static_exit_2 | 91.313 | 135700.0 | False |
| static_exit_3 | 92.501 | 156084.0 | True |

## Interpretation and limits

The evaluated family spans 81,456 to 156,084 FLOPs/window. 17 of 30 configurations are dominated in this sample (equal-coordinate ties are retained).

See baseline_comparison.json for accuracy differences (percentage points) and compute reductions relative to the fixed full-depth path. No policy is chosen using these test differences.

The three static references use the frozen baseline checkpoint and skip unused heads. They are fixed-depth ablations, not separately trained or tuned static CNNs. A stronger independently trained static comparison and a compression experiment remain project work.

The included checkpoint is exploratory and single-seed; its documentation discloses prior inspection of test subjects. This workflow prevents new test-threshold tuning but cannot make those subjects a fresh holdout. Small differences need repeated runs/uncertainty analysis.

Synthetic or smoke-run results are only software checks, not evidence of HAR performance. Changing training, compression, preprocessing or calibration requires a new validation manifest.

## Descriptive matched-accuracy comparison

Among the evaluated adaptive configurations, `normal_0.98_0.8` is closest in accuracy to the full-depth static reference: -0.034 percentage points, with 30.51% fewer calculated FLOPs. This is a description of the plotted results, not a test-selected deployment policy or evidence of statistical equivalence. Negative compute reduction means more work.
