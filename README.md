# Adaptive wearable activity recognition

A compact PyTorch project for wearable human activity recognition: subject-aware
data preparation, one three-exit residual CNN, and confidence-based adaptive
inference. The objective is to trade computation for recognition quality by
stopping confident windows at an early exit.

## Model architecture

One three-exit residual 1D CNN with 2,486 parameters (9,944 bytes in float32).
Input is a normalized window of shape `(B, 9, 128)`: nine inertial channels
over 128 time steps. Each exit returns `(B, 6)` class logits.

```text
(B, 9, 128)
  -> Stem: Conv1d k7 s2 (9->4) + BN + ReLU                 (B, 4, 64)
  -> Stage 1: 4 residual blocks, 4 ch                      (B, 4, 64)  -> Exit 1
  -> Stage 2: 3 residual blocks, 7 ch, first stride 2      (B, 7, 32)  -> Exit 2
  -> Stage 3: 2 residual blocks, 7 ch, first stride 2      (B, 7, 16)  -> Final
```

- **Residual block:** Conv1d k3 + BN + ReLU, Conv1d k3 + BN, add skip, ReLU.
  The skip is a 1×1 Conv + BN when channels or stride change. Convolutions
  have no bias.
- **Exit head:** global average pooling, dropout (0.40 / 0.40 / 0.10), Linear to six classes.
- **Training:** all exits train jointly with cross-entropy weighted 0.10 / 0.20 / 0.70.

| Component | Parameters | Cumulative parameters | FLOPs added (segment + head) | Total FLOPs, fixed depth | Total FLOPs, adaptive path |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stem | 260 | 260 | 32,256 | — | — |
| Stage 1 + Exit 1 | 448 + 30 | 738 (29.7%) | 49,152 + 48 | 81,456 | 81,456 |
| Stage 2 + Exit 2 | 945 + 48 | 1,731 (69.6%) | 54,208 + 84 | 135,700 | 135,748 |
| Stage 3 + Final | 707 + 48 | 2,486 (100%) | 20,384 + 84 | 156,084 | 156,216 |

Both total columns are the full cost of one window from the input up to that
exit, not the increment. A fixed-depth run skips earlier heads; an adaptive
path also pays for every head it visits. The stem and Stage 1 alone are 52% of
full-depth compute. FLOPs = 2 × Conv1d/Linear MACs, excluding BN, activations,
pooling and residual additions: a compute proxy, not measured latency or energy.

### Exit policy

Each exit's confidence is `softmax(logits / T).max()`, using per-exit temperatures
(0.171 / 0.340 / 0.609) fitted on validation. A window stops at the first exit
whose confidence reaches its threshold; the remaining batch is compacted, so
later stages really run only on unresolved windows.

- **Normal mode:** exit at confidence ≥ 0.98 (Exit 1) or ≥ 0.80 (Exit 2), otherwise Final.
- **Low-power mode:** confidence-based Exit 1, otherwise stop at Exit 2. Stage 3 never runs.

## Selected checkpoint

The included checkpoint is [models/adaptive_har.pt](models/adaptive_har.pt)
(epoch 41, seed 42). Measured accuracies:

| Fold | Exit 1 | Exit 2 | Final |
| --- | ---: | ---: | ---: |
| Validation | 91.02% | 93.21% | 94.64% |
| Test | 90.53% | 91.31% | 92.50% |

The test gains are 0.78 and 1.19 percentage points. Final accuracy reaches the
approximately 92% goal; the first gap remains below the strict one-point target.
See [the model record](docs/model.md) for exact results, calibration, selected
policies and reproducibility limits. Only this model is supported.

## Setup

Python 3.10 or later is required; the selected run used Python 3.12 and CPU.

```bash
git clone https://github.com/gabrielebnc/bde-adaptive-har.git
cd bde-adaptive-har
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones)
(Reyes-Ortiz et al., 2013, [DOI: 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), CC BY 4.0):
30 subjects, six activities, raw inertial windows (not the 561 precomputed features).

```bash
python -m src.data --download --data-dir data   # download, extract and validate
python -m src.data --data-dir data              # validate an existing copy
```

Expected layout (git-ignored):

```text
data/UCI HAR Dataset/
├── train/  Inertial Signals/ (9 files), y_train.txt, subject_train.txt
└── test/   Inertial Signals/ (9 files), y_test.txt, subject_test.txt
```

Channels: body acceleration x/y/z, body gyroscope x/y/z, total acceleration x/y/z.
Labels 0–5: walking, upstairs, downstairs, sitting, standing, laying. Splits are
subject-disjoint: 5,392 training, 1,960 validation (subjects 17, 21, 25, 26, 30)
and 2,947 official test windows. Normalization is fitted on training only and
stored in the checkpoint.

## Evaluate the included checkpoint

```bash
python evaluate.py --data-dir data --split validation --device cpu
python evaluate.py --data-dir data --split test --device cpu
```

These commands default to `models/adaptive_har.pt` and write generated reports
under `runs/adaptive_har/`, leaving the included model unchanged.
Reports contain accuracy, macro F1, per-class metrics and confusion matrices.
Use `--checkpoint` for a retrained checkpoint and `--output-dir` for custom
report locations.

## Adaptive inference

The included model's validation-selected thresholds are 0.98 / 0.80 in normal
mode, and 0.98 at Exit 1 in low-power mode:

```bash
python adaptive_evaluate.py --data-dir data --split test --mode normal --threshold-1 0.98 --threshold-2 0.80 --device cpu
python adaptive_evaluate.py --data-dir data --split test --mode low-power --threshold-1 0.98 --device cpu
```

| Mode | Test accuracy | Macro F1 | Average stages | Estimated MACs/window |
| --- | ---: | ---: | ---: | ---: |
| Normal | 92.47% | 92.59% | 1.579 | 54,231 |
| Low-power | 91.35% | 91.51% | 1.448 | 52,887 |

These thresholds belong to this checkpoint. For a new checkpoint, select them
on validation, then freeze them before test evaluation:

```bash
python adaptive_evaluate.py --checkpoint runs/adaptive_har_retrain/best.pt --data-dir data --split validation --mode normal --sweep
python adaptive_evaluate.py --checkpoint runs/adaptive_har_retrain/best.pt --data-dir data --split validation --mode low-power --sweep
```

Test-set sweeps are disabled. MACs estimate Conv/Linear computation, not measured
latency, power consumption or battery life.

## Live resource-shock demonstration

```bash
python -m sim
```

A desktop window compares the battery-aware model, fixed normal-mode inference,
and fixed full-depth inference on the same shuffled test windows. Three stacked
graphs show compute and cumulative accuracy, with exit usage and latency metrics.
The battery slider enters low-power at 35%, recovers normal at 40%, and forces
Exit 1 below 20%. No new model or training is involved. See [sim/README.md](sim/README.md)
for the controller, controls, headless replay, exports and measurement limitations.

## Pareto frontier evaluation

Generate accuracy-versus-FLOPs plots with static fixed-depth
references and validation-selected adaptive policies:

```bash
python pareto_evaluate.py --split validation --device cpu --output-dir runs/pareto_v1/validation
python pareto_evaluate.py --split test --device cpu --policies runs/pareto_v1/validation/policies.json --output-dir runs/pareto_v1/test
```

Outputs include PNG/SVG plots, all operating points, a frozen policy manifest,
baseline comparisons and provenance. Test thresholds cannot be swept; a changed
checkpoint requires new validation selection. Static references skip unused
heads and use `--baseline-checkpoint` (the original included model by default),
which should stay fixed across later experiments. See [the evaluation protocol](docs/pareto.md)
for compute conventions, selection rules, outputs and limitations.

The included checkpoint's [test frontier](docs/results/pareto/test/pareto.png)
and [test report](docs/results/pareto/test/report.md) are saved alongside the
[validation sweep](docs/results/pareto/validation/pareto.png) and frozen policy
manifest. The normal 0.98/0.80 policy uses 30.51% fewer calculated FLOPs than
the fixed full-depth reference, with a 0.034 percentage-point test accuracy
decrease. These remain exploratory, single-seed results.

## Reproduce training

After cloning the repository and completing Setup, rebuild the current model
with one command:

```bash
python scripts/reproduce_model.py
```

This downloads UCI HAR if needed, trains from random initialization using
[the saved recipe](models/adaptive_har.recipe.json), fits confidence temperatures,
selects adaptive policies on validation, evaluates on test, and checks the
rebuilt weights against the recorded model fingerprint. It does not read
`models/adaptive_har.pt`, private documents, or previous runs.

The rebuilt checkpoint is `runs/adaptive_har_retrain/best.pt`. Metrics,
selected policies, training history and `reproduction_check.json` are saved
beside it. Evaluate it with:

```bash
python evaluate.py --checkpoint runs/adaptive_har_retrain/best.pt --split test --device cpu
```

For the tested rebuild environment, use Python 3.12 and install the pinned
dependencies before running the command:

```bash
python -m pip install -r requirements-reproduce.txt
python scripts/reproduce_model.py --require-exact
```

`--require-exact` makes the command fail if the weights, calibration or selected
epoch differ from the reference. Other platforms or library versions can train
successfully with different results; the report records whether they match.
To repeat training, choose a fresh directory with `--output-dir runs/my_rebuild`.
Use `--data-dir /path/to/data` for a dataset elsewhere. The full manual command
and reproducibility limits are in [the model record](docs/model.md#rebuild-from-source).
Training never uses test labels to select checkpoints or thresholds.

Training writes the checkpoint, settings, split metadata, model summary,
histories, plots and calibration diagnostics to its output directory.
Existing checkpoints are protected from overwrite; choose a fresh output
directory for another run. The included `models/adaptive_har.pt` is never
overwritten by training.

## Verification

```bash
python -m unittest discover -s tests -v
python -m tests.smoke_train --output-dir runs/offline_smoke
```

Tests cover checkpoint integrity/loading, the fixed architecture and exit budgets,
data split integrity, training-only normalization, gradients, calibration,
metrics, the saved rebuild recipe and model fingerprint, and actual conditional
execution. When UCI HAR is available, tests
also reproduce the included checkpoint's saved validation/test metrics.
The offline CLI smoke test uses explicitly labeled synthetic data; its numbers
are not trained-model performance.
