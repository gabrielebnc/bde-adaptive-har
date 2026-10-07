# Adaptive wearable activity recognition

A compact PyTorch project for wearable human activity recognition: subject-aware
data preparation, one three-exit residual CNN, and confidence-based adaptive
inference. The objective is to trade computation for recognition quality by
stopping confident windows at an early exit.

## Selected model

The included checkpoint is [models/adaptive_har.pt](models/adaptive_har.pt).
It has 2,486 parameters and these measured accuracies:

| Fold | Exit 1 | Exit 2 | Final |
| --- | ---: | ---: | ---: |
| Validation | 91.02% | 93.21% | 94.64% |
| Test | 90.53% | 91.31% | 92.50% |

The test gains are 0.78 and 1.19 percentage points. Final accuracy reaches the
approximately 92% goal; the first gap remains below the strict one-point target.
The checkpoint is a regular project file; Git LFS is not required.
See [the model record](docs/model.md) for architecture, exact results, calibration,
selected adaptive policies, and reproducibility limits.

Only this model is supported. Public documentation is under `docs/`;
`docs/private/` is ignored. Datasets, generated runs and local environments
are also ignored.

## Setup

Python 3.10 or later is required; the selected run used Python 3.12 and CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset download and location

Run this command from the repository root, with the virtual environment activated:

```bash
python -m src.data --download --data-dir data
```

Use [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones),
by Reyes-Ortiz, Anguita, Ghio, Oneto and Parra (2013),
[DOI: 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), licensed CC BY 4.0.
The downloader retains and validates the complete archive and extracts all files,
including any nested dataset ZIP. It records the archive SHA-256 and source URL
in `data/download.json`. The code consumes inertial windows, not the 561 feature
vectors.

The command downloads and extracts the dataset automatically into the repository's
`data/` directory. The required layout is:

```text
data/
└── UCI HAR Dataset/
    ├── train/
    │   ├── Inertial Signals/     # all nine *_train.txt signal files
    │   ├── y_train.txt
    │   └── subject_train.txt
    └── test/
        ├── Inertial Signals/     # all nine *_test.txt signal files
        ├── y_test.txt
        └── subject_test.txt
```

Keep the folder name `UCI HAR Dataset` and its train/test subdirectories intact.
Do not flatten the extracted files or use only the 561-feature tables.
The full `data/` directory is git-ignored; it should not be pushed to GitHub.

For a manual download, use the Download button on the UCI page linked above,
save the complete ZIP as `data/uci_har_complete.zip`, and run the preparation
command above. It also extracts the inner dataset ZIP when the download is wrapped
in an outer ZIP. Alternatively, download the same archive from a terminal:

```bash
mkdir -p data
curl -L --fail 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip' -o data/uci_har_complete.zip
python -m src.data --download --data-dir data
```

If the archive is already fully extracted at `data/UCI HAR Dataset/`, validate
and prepare it without downloading again:

```bash
python -m src.data --data-dir data
```

For a dataset outside this repository, pass its parent directory through
`--data-dir` to preparation, training and evaluation. For example,
`--data-dir /path/to/datasets` expects
`/path/to/datasets/UCI HAR Dataset/`.

Each input is a float32 window shaped `(9, 128)`; batches are `(B, 9, 128)`.
Channel order is body acceleration x/y/z, body gyroscope x/y/z, then total
acceleration x/y/z. Labels are 0–5: walking, walking upstairs, walking downstairs,
sitting, standing and laying.

Official train/test subjects remain separate. Validation holds out whole
training subjects; normalization is fitted on the training fold only.
Evaluation restores the checkpoint's subject lists and normalization rather than
refitting. Dataset files are not committed.

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

Normal mode exits at the first accepted calibrated confidence; unresolved windows
reach the final exit. Low-power mode never executes Stage 3. Inference compacts
the remaining batch after each exit and preserves original sample order.

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

## Pareto frontier evaluation

Generate the assignment's accuracy-versus-FLOPs plots with static fixed-depth
references and validation-selected adaptive policies:

```bash
python pareto_evaluate.py --split validation --device cpu --output-dir runs/pareto_v1/validation
python pareto_evaluate.py --split test --device cpu --policies runs/pareto_v1/validation/policies.json --output-dir runs/pareto_v1/test
```

Outputs include PNG/SVG plots, all operating points, a frozen policy manifest,
baseline comparisons and provenance. Test thresholds cannot be swept; a changed
checkpoint requires new validation selection. Static references skip unused
heads and use `--baseline-checkpoint` (the original included model by default),
which should stay fixed across later experiments. These are preliminary comparisons:
compression and an independently trained static baseline remain future work.
See [the evaluation protocol](docs/pareto.md) for the FLOPs convention,
selection rules, reproduction instructions and assignment limitations.

The included checkpoint's [test frontier](docs/results/pareto/test/pareto.png)
and [test report](docs/results/pareto/test/report.md) are saved alongside the
[validation sweep](docs/results/pareto/validation/pareto.png) and frozen policy
manifest. The normal 0.98/0.80 policy uses 30.51% fewer calculated FLOPs than
the fixed full-depth reference, with a 0.034 percentage-point test accuracy
decrease. These remain exploratory, single-seed results.

## Reproduce training

```bash
python train.py --data-dir data --output-dir runs/adaptive_har_retrain --device cpu
python scripts/evaluate_run.py --run-dir runs/adaptive_har_retrain --data-dir data --device cpu --threads 2
```

Defaults match the selected recipe: seed 42, AdamW, learning rate 0.001,
weight decay 0.0001, batch size 64, cosine decay over up to 100 epochs,
patience 30, two CPU threads, dropout 0.40 / 0.40 / 0.10, and loss weights
0.10 / 0.20 / 0.70.

Checkpoint selection first prefers both adjacent validation accuracy gains
being at least one point, then maximizes final validation accuracy and minimizes
weighted cross-entropy. If no epoch qualifies, the best final-accuracy candidate
is retained and the unmet target is recorded. No test labels select an epoch.
One positive temperature per exit is fitted on validation; scaling changes
confidence, not predicted classes.

Training writes the checkpoint, settings, split metadata, model summary,
histories, plots and calibration diagnostics to its output directory.
Existing checkpoints are protected from overwrite; choose a fresh output
directory for another run. The included `models/adaptive_har.pt` is never
overwritten by training. Results may differ across devices and library versions.

## Verification

```bash
python -m unittest discover -s tests -v
python -m tests.smoke_train --output-dir runs/offline_smoke
```

Tests cover checkpoint integrity/loading, the fixed architecture and exit budgets,
data split integrity, training-only normalization, gradients, calibration,
metrics, and actual conditional execution. When UCI HAR is available, tests
also reproduce the included checkpoint's saved validation/test metrics.
The offline CLI smoke test uses explicitly labeled synthetic data; its numbers
are not trained-model performance.
