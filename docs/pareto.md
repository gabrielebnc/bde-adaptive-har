# Accuracy versus FLOPs evaluation

This workflow addresses assignment Sections 5, 6.4 and 9: three or more compute
levels, consistent calculated FLOPs, static comparisons, and validation-only
threshold selection before final evaluation. It does not add compression or
claim that the complete assignment is finished.

## Included checkpoint results

The saved [validation results](results/pareto/validation/results.json)
contain 113 configurations: 100 normal threshold pairs, ten low-power
thresholds and three static depths. Validation selection freezes 27 adaptive
policies plus the three static references. All 30 are evaluated on the official
test subjects, including 17 configurations dominated on that sample.

![Test accuracy versus calculated FLOPs](results/pareto/test/pareto.png)

The frozen normal 0.98/0.80 policy obtains 92.467% test accuracy at approximately
108,462 FLOPs/window. The full-depth static reference obtains 92.501% at
156,084 FLOPs/window: 30.51% less calculated compute for a 0.034 percentage-point
accuracy decrease. This is a descriptive near-matched-accuracy comparison;
the difference is one test window, not evidence of statistical equivalence.
The middle fixed-depth reference is dominated by several adaptive policies.
Low-power policies are retained because of their hard Stage-3 exclusion,
even where normal policies dominate them on average accuracy and FLOPs.

See the [test report](results/pareto/test/report.md),
[operating-point table](results/pareto/test/operating_points.csv),
[validation plot](results/pareto/validation/pareto.png) and
[policy manifest](results/pareto/validation/policies.json). The original
checkpoint and baseline hashes are recorded in the JSON outputs. These saved
figures are a before-compression reference, not a completed compression study.

## Run

Install `requirements.txt` and prepare UCI HAR as described in the README. From
the repository root (on Windows, activate `.venv\Scripts\Activate.ps1`):

```bash
python pareto_evaluate.py --split validation --device cpu --output-dir runs/pareto_v1/validation
python pareto_evaluate.py --split test --device cpu --policies runs/pareto_v1/validation/policies.json --output-dir runs/pareto_v1/test
```

Both commands default to `models/adaptive_har.pt`. For a retrained adaptive model,
pass the same new `--checkpoint` to both commands and use a new output directory.
The separate `--baseline-checkpoint` defaults to the original
`models/adaptive_har.pt`: keep it fixed across experiments. Its hash is also bound
to the manifest. Both checkpoints must use identical subject splits, channels,
labels and dataset sizes; each restores its own training normalization.
Existing nonempty output directories are protected from overwrite. No training
or checkpoint mutation happens during evaluation.

Validation evaluates a 10-by-10 grid of normal-mode thresholds and ten low-power
thresholds. Override these with comma-separated `--thresholds-1` and
`--thresholds-2` on validation only. The finite grid is a sampled policy family,
not an exhaustive search of every real-valued threshold. A threshold of 1 is
still a confidence comparison (`>=`); floating-point softmax may equal 1, so
it is **not** used to implement forced-depth static references.

Every candidate appears in the validation table and plot. The manifest keeps
all non-dominated validation policies **within each mode**, preserving a
low-power family even if normal mode dominates it globally. Equal accuracy/cost
coordinates use the first policy in sorted grid order. Three explicit static
depths are always retained. Static references do not eliminate adaptive policies
during selection. Selected cached validation estimates are verified against
actual conditional execution; a routing discrepancy stops the run.

Test evaluates every frozen manifest entry, including entries that become
dominated on test. It never searches thresholds or filters the evaluated family
using test accuracy. The test frontier is a descriptive highlight over this
fixed family, not a new deployment-policy selection. Choose deployment budgets
on validation. Manifests are tied to checkpoint and evaluation-code hashes;
changing either requires a fresh validation run. Hashes provide traceability,
not protection against deliberate manifest editing.

## Outputs

Each directory contains:

- `pareto.png` and `pareto.svg`: accuracy (%) versus average FLOPs per window,
  with normal, low-power and fixed-depth configurations distinguished.
- `operating_points.csv`: every evaluated point, exit frequencies, macro F1,
  accuracy, average stages, MACs, FLOPs and a non-dominated flag.
- `results.json`: metrics plus checkpoint/code hashes, data splits and saved
  normalization, temperatures, device/runtime settings and detailed classification
  reports for executed policies.
- `policies.json`: selected validation policies; copied unchanged into test output.
- `baseline_comparison.json`: accuracy change in percentage points and compute
  reduction relative to the static full-depth reference, for each adaptive point.
- `report.md`: readable table, methodology and interpretation limits.

A point is dominated if another evaluated point has no greater cost and no
lower accuracy, with at least one strict improvement. Identical coordinates
remain non-dominated ties in results. The dashed frontier joins unique
non-dominated coordinates only as a visual guide; intermediate configurations
are not measured and are not guaranteed attainable.

## Compute methodology and static baseline

The same hook-based Conv1d/Linear counter is used for actual static and adaptive
execution. One multiply-accumulate counts as two FLOPs. It uses executed output
shapes, including the actual remaining batch size after each exit. Validation
sweep estimates use the existing `model_summary` per-path MAC counts and exact
integer exit counts. For adaptive routing:

`average FLOPs = 2 * sum(number exiting at i * cumulative MACs to i) / N`.

Cumulative adaptive costs include all visited exit heads. The counter excludes
bias additions, BatchNorm, activation functions, residual additions, pooling,
softmax, routing and data movement. This is a consistently calculated
Conv/Linear FLOPs proxy under the assignment's calculation allowance; it is
not total runtime work, measured latency, energy, or battery life. Do not mix
these values with another profiler's convention without recalculating all points.

The static references execute exactly one fixed depth and its final classifier,
skipping unused intermediate heads and all confidence decisions. Their weights
come from the unchanged baseline checkpoint. For the included architecture the costs
are 81,456 / 135,700 / 156,084 FLOPs per window. The latter two are slightly below
the old cumulative adaptive-path counts because unused heads no longer execute.
These are transparent fixed-depth ablations, including a non-adaptive full-depth
baseline. They are **not independently trained, well-tuned static CNNs**. Such a
baseline would strengthen the assignment comparison and should be frozen before
comparing subsequent adaptive improvements.

## Interpretation and assignment limits

Compare points at similar accuracy or compute, using the full-depth comparison
table for percentage-point accuracy losses and percentage compute reductions.
Do not claim improvement from an arbitrary point-to-point comparison with a
large accuracy difference. The frontier is empirical over the evaluated family,
not proof of a globally optimal model or statistically significant improvement.

The current network has early exits but no implemented compression technique.
Add the chosen compression experiment and evaluate a new checkpoint using this
same protocol. Keep the present curve as the before-compression reference.
Training/KD/calibration changes may move adaptive FLOPs even without architecture
changes, because exit frequencies change. Structured pruning changes path costs.
Quantization does not by itself reduce the number of arithmetic operations;
report precision and memory/latency separately when adding quantized models.

The selected checkpoint's model record discloses single-seed exploratory results
and prior test-subject inspection. This protocol prevents additional threshold
tuning on test but cannot undo that history. Use a fresh holdout where available,
or disclose the limitation. Repeated training seeds and subject-level uncertainty
estimates would strengthen conclusions, especially for tiny accuracy differences.
Synthetic smoke outputs are labelled and must never be used as project results.

## Verification

```bash
python -m unittest discover -s tests -v
```

The Pareto tests cover domination and ties, per-mode selection, exact forced
depths, skipped-head accounting, mixed-batch execution costs, manifest mismatch
rejection, test-sweep rejection, overwrite protection, and a complete synthetic
validation-to-test CLI run that produces plots and reports.
