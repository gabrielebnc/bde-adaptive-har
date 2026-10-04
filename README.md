# Adaptive wearable activity recognition

A first version in Python/PyTorch: subject-aware data preparation, a three-exit
residual CNN, joint training, per-exit evaluation, and confidence-based adaptive
inference. The scope ends at model experiments and inference.

## Environment

Python 3.10 or later (tested with Python 3.12). Use a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project-local `.venv` was populated using cached wheels in this workspace.
All commands below use the activated environment; `.venv/bin/python` is an
equivalent explicit interpreter. Device selection is CUDA, then MPS, then CPU;
override it with `--device cpu`, `--device mps`, or `--device cuda`.
Python, NumPy, PyTorch, and loader generators are seeded (default 42).
Deterministic algorithms are requested, with warnings for unsupported kernels.
Results can still differ across devices, library versions, and hardware.

## Dataset preparation

Use [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones),
by Reyes-Ortiz, Anguita, Ghio, Oneto, and Parra (2013),
[DOI: 10.24432/C54S4K](https://doi.org/10.24432/C54S4K), licensed CC BY 4.0.
The complete archive is required. The code reads the provided inertial windows,
which UCI has already filtered and segmented, rather than the 561 feature vectors.

```bash
python -m src.data --download --data-dir data
```

This downloads the entire ZIP, validates its CRC, retains it, and extracts **all**
files. If the UCI download contains a nested `UCI HAR Dataset.zip`, that complete
ZIP is also extracted. A SHA-256 digest and source URL are saved to
`data/download.json`. There is no runtime dependency on `ucimlrepo`.

If network access is restricted in the coding environment, download in a normal
terminal instead:

```bash
mkdir -p data
curl -L --fail 'https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip' -o data/uci_har_complete.zip
python -m src.data --download --data-dir data
```

The preparation command reuses the local ZIP and extracts it. Alternatively,
manually extract the dataset to `data/UCI HAR Dataset/` and omit `--download`.

Signals come from `train/Inertial Signals/` and `test/Inertial Signals/`, in this
fixed channel order: `body_acc_x`, `body_acc_y`, `body_acc_z`, `body_gyro_x`,
`body_gyro_y`, `body_gyro_z`, `total_acc_x`, `total_acc_y`, `total_acc_z`.
Each float32 example has shape `(9, 128)`. Batch shape is `(B, 9, 128)`.
Labels are converted from 1–6 to int64 values 0–5:

| Label | Activity |
| --- | --- |
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

Official train/test subjects stay separate. Validation holds out whole subjects
from `subject_train.txt`: a seeded permutation chooses `ceil(0.2 * subjects)`
by default. Set `--val-fraction` or supply an explicit list, for example
`--val-subjects 1 3 5 6`. Only official training subjects may be held out.
Windows from a subject never appear in both the training and validation folds.

Each channel's mean and population standard deviation are calculated across
training-fold windows and time steps only. The same nine means and standard
deviations normalize validation/test data. Constant-channel standard deviations
are floored at `1e-6`. Subject IDs, channel order, counts, and normalization are
saved in the checkpoint; evaluation restores these instead of fitting again.
The official test fold is loaded for shape/split checks but is never used for
loss optimization, checkpoint selection, calibration, or threshold selection.

## Models: current Tiny 2.5k and retained experiments

The default `tiny_25k` is a **2,486-parameter** three-exit residual CNN. Cumulative
exit budgets approximate the requested 30% / 70% / 100%. Both the preceding 1k
and 5k experiments, their fine-tuning attempts, checkpoints, reports and original
plots are retained unchanged. Micro remains available without saved runs.

| Architecture | Stem channels | Stage channels | Blocks per stage | Exit 1 parameters | Exit 2 parameters | Full parameters |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Tiny 2.5k (`tiny_25k`, default) | 4 | 4 / 7 / 7 | 4 / 3 / 2 | 738 (29.69%) | 1,731 (69.63%) | 2,486 |
| Tiny 1k (`tiny_1k`, retained) | 1 | 4 / 5 / 6 | 2 / 2 / 1 | 295 (29.74%) | 686 (69.15%) | 992 |
| Tiny 5k (`tiny_5k`, retained) | 7 | 7 / 10 / 15 | 3 / 3 / 1 | 1,469 (29.88%) | 3,455 (70.28%) | 4,916 |
| Micro (`micro`) | 4 | 4 / 8 / 16 | 2 / 2 / 2 | 514 | 1,352 | 4,430 |

All accept `(B, 9, 128)` and return three `(B, 6)` logits tensors. Residual
blocks use two bias-free kernel-3 convolutions with BatchNorm/ReLU and projected
skip connections when channels or resolution change. The kernel-7 stem has
stride 2; Stages 2 and 3 each downsample by two. Heads use global average pooling,
dropout (0.20 / 0.25 / 0.30), and a linear classifier. Architectures are fixed
before training, with no runtime width switching.

| Tiny 2.5k component | Output shape | Cumulative parameters | Conv/Linear MACs/window |
| --- | --- | ---: | ---: |
| Input | `(B, 9, 128)` | — | — |
| Stem | `(B, 4, 64)` | — | — |
| Stage 1 / Exit 1 | `(B, 4, 64)` / `(B, 6)` | 738 | 40,728 |
| Stage 2 / Exit 2 | `(B, 7, 32)` / `(B, 6)` | 1,731 | 67,874 |
| Stage 3 / Final | `(B, 7, 16)` / `(B, 6)` | 2,486 | 78,108 |

Counts include earlier heads. MACs exclude BatchNorm, pooling, activations,
softmax and routing; estimated FLOPs are twice MACs. Parameter fractions are not
energy fractions. No device latency, power, or battery-life measurement is claimed.

See the [2.5k experiment record](docs/tiny_25k_model.md), the retained
[1k record](docs/tiny_1k_model.md), and [5k record](docs/tiny_model.md).

## Train

```bash
python train.py --data-dir data --model-size tiny_25k --output-dir runs/tiny_25k_reproduce
```

Use `--download` if data preparation has not yet run. Defaults are AdamW, learning
rate `1e-3`, weight decay `1e-4`, batch size 64, up to 100 epochs, early-stopping
patience 12, and cosine learning-rate decay. Joint loss is:

```text
0.20 * CE(exit1, y) + 0.30 * CE(exit2, y) + 0.50 * CE(final, y)
```

`best.pt` minimizes this same **weighted validation loss across all exits**.
The training-fold class counts are printed and saved. Ordinary cross-entropy is
used; class weighting should only be introduced after real-data inspection and
training results demonstrate a material imbalance issue.

After loading the best checkpoint, one positive temperature per exit is fitted
on the full validation fold by minimizing cross-entropy, then saved in `best.pt`.
Confidence uses `softmax(logits / temperature).max()`. Calibration records
validation NLL before and after; it falls back to temperature 1 if fitting worsens
NLL. Scaling does not change each exit's predicted class. It improves the NLL
objective, but does not guarantee perfectly calibrated probabilities on unseen
subjects. No calibration uses test labels.

Important settings are exposed through `python train.py --help`, including seed,
subject split, dropout rates, loss weights, optimizer settings, epoch limits,
threads, workers, and calibration iterations. Existing checkpoints are protected
from accidental overwrite; use a new output directory for another experiment.

Artifacts in the run directory:

- `best.pt`: weights, model/training settings, split metadata, normalization,
  optimizer/scheduler state, validation temperatures, and best epoch.
- `config.json`, `data_metadata.json`, `model_summary.json`.
- `history.json`, `history.csv`: sample-weighted train/validation losses and
  accuracy for all exits, epoch, learning rate, and processed sample counts.
- `history.png`: loss and accuracy curves.
- `training_summary.json`: best epoch, duration, device, calibration diagnostics,
  and whether training/validation was truncated for a smoke run.

## Evaluate every exit

```bash
python evaluate.py --checkpoint runs/tiny_25k/best.pt --data-dir data --split test
```

The default evaluates the separate official test subjects. Use `--split
validation` for validation results. Each exit reports accuracy, macro F1,
per-class precision/recall/F1/support, and a 6×6 confusion matrix, together with
full-model and cumulative executed parameter counts and approximate computation.
Zero-support class metrics are zero; macro F1 always includes all six classes.
Confusion matrix rows are true labels and columns are predictions.

Results are saved in `runs/tiny_25k/evaluation_test/metrics.json`, `per_class.csv`,
and `confusion_matrices.png` (or a custom `--output-dir`). Evaluation also checks
the saved subject lists, dataset sizes, class names, and channel order.

## Select adaptive thresholds on validation

```bash
python adaptive_evaluate.py --checkpoint runs/tiny_25k/best.pt --data-dir data --split validation --mode normal --sweep
python adaptive_evaluate.py --checkpoint runs/tiny_25k/best.pt --data-dir data --split validation --mode low-power --sweep
```

The sweep grid is configurable, for example `--thresholds-1 0,0.7,0.8,0.95,1
--thresholds-2 0,0.6,0.8,0.95,1`. These are **candidate values**, not final policy
recommendations. The default grid explores a broad range including boundary
values. For every combination, CSV/JSON reports accuracy, macro F1, percent
leaving at each exit, average stages executed, and average estimated MACs.
A plot shows validation accuracy versus average stages, colored by macro F1.

Normal mode returns Exit 1 when calibrated confidence is at least threshold 1;
otherwise it evaluates Stage 2 and returns Exit 2 if confidence is at least
threshold 2; otherwise it evaluates Stage 3 and returns the final exit.
Low-power mode applies threshold 1, then always returns Exit 2 for remaining
windows. Stage 3 never runs, and threshold 2 does not affect this mode.

Sweeps use cached all-exit logits to compare policies efficiently. Their exit
fractions, stages, and MACs describe estimated policy execution, not a timed
adaptive run. Choose an acceptable validation accuracy/F1 versus computation
tradeoff and freeze the thresholds before running the official test evaluation.
Test-set sweeps are disabled. Calibration, checkpoint selection, and sweeps share
the validation fold. Test subjects remain separate from training; these subjects
have already been inspected in earlier experiments, so repeated test comparisons
are not a fresh independent confirmation.

To evaluate a frozen policy, fill in thresholds selected from the sweep:

```bash
python adaptive_evaluate.py --checkpoint runs/tiny_25k/best.pt --data-dir data --split test --mode normal --threshold-1 "$T1" --threshold-2 "$T2"
python adaptive_evaluate.py --checkpoint runs/tiny_25k/best.pt --data-dir data --split test --mode low-power --threshold-1 "$T1_LOW_POWER"
```

Set those shell variables to your selected numeric values first. Fixed-policy
evaluation uses `AdaptiveHAR.adaptive_forward()` with actual conditional execution:
only unresolved samples reach later stages. Outputs preserve original batch order
and contain calibrated logits, predictions, confidences, and 1-based exit IDs.
Call `model.eval()` before adaptive inference. Supply checkpoint temperatures and
normalized input windows; policy thresholds are explicit, with no assumed final
defaults. Confidence acceptance uses `>=`. Average stages is the mean exit ID.
Fixed-policy reports are saved under `adaptive_test_normal/` or
`adaptive_test_low-power/` as `policy_metrics.json`.

## Tests and smoke training

Tests use Python's standard `unittest`; no separate test runner is needed:

```bash
python -m unittest discover -s tests -v
```

Tests check model shapes/size, subject split integrity, label conversion,
training-only normalization, backward gradients at every exit, calibration,
metrics, nested archive extraction, batch compaction/order, and actual skipping
of later stages. A file-based synthetic batch test always runs. A real UCI batch
forward/backward smoke test also runs when the official data is present, otherwise
it is explicitly skipped.

Run a short training smoke test on the real dataset after downloading it:

```bash
python train.py --data-dir data --output-dir runs/uci_smoke --epochs 2 --batch-size 8 --max-train-batches 2 --max-val-batches 1
```

For a fully offline CLI smoke check:

```bash
python -m tests.smoke_train --output-dir runs/offline_smoke
```

This creates small synthetic UCI-format files and runs two truncated training
epochs, best-checkpoint calibration, per-exit test evaluation, validation sweeps
in both modes, and fixed adaptive policies. It asserts that reports/plots exist
and that routing boundaries work. Synthetic fixture reports carry
`data_source: synthetic_test_fixture`; truncated runs carry `smoke_run: true`.
Use a fresh output directory when repeating either smoke command.

The tests include an actual UCI batch forward/backward pass when data is present.
The current Tiny has a two-epoch truncated real-data smoke run under
`runs/tiny_25k_smoke/`; those metrics are correctness checks, not performance.

## Fine-tune the final exit

If the final classifier trails Exit 2, fine-tune Stage 3 and the final head in a
fresh directory:

```bash
python finetune_final.py --checkpoint runs/tiny_25k/best.pt --data-dir data --output-dir runs/tiny_25k_finetuned_reproduce --device cpu --threads 2 --lr 0.0001 --epochs 30 --patience 8
```

The prefix weights and BatchNorm buffers are frozen, and earlier dropout stays
in evaluation mode. The source checkpoint is included as an epoch-zero candidate;
selection uses weighted validation loss. Only final-exit CE supplies training
gradients. Earlier calibration temperatures are preserved; a changed final head
is recalibrated on validation. The source checkpoint is never overwritten.
Fine-tuning reports label the different training/validation loss objectives.
Re-evaluate all exits and reselect adaptive thresholds from validation after
fine-tuning. Equal or increasing test accuracy across exits is not guaranteed.

Each experiment record states whether fine-tuning was triggered and reports its
actual outcome. The 2.5k, 1k and 5k checkpoints, metrics, sweeps, plots,
and smoke/test records are retained. The shared dataset and virtual environment are preserved.

If early exits perform poorly, knowledge distillation remains a future option.

## Completed reports and plots

Evaluate all exits, sweep validation thresholds, freeze the best validation policy
per mode, and execute it on test subjects with:

```bash
python scripts/evaluate_run.py --run-dir runs/tiny_25k --device cpu --threads 2
```

This helper uses validation accuracy for policy selection, breaking ties with
lower estimated MACs, then higher macro F1. It does not sweep test thresholds.
Generate current Tiny joint/fine-tuning plots with:

```bash
python scripts/plot_tiny.py --run-dir runs/tiny_25k --output-dir docs/figures/tiny_25k
```

The new plots are saved as PNG/SVG in `docs/figures/tiny_25k/`, alongside exact
plotted values in CSV/JSON. The original 1k figures remain in `docs/figures/tiny_1k/`, and 5k figures in `docs/figures/`.
See [Tiny 2.5k's actual outcomes](docs/tiny_25k_model.md). Each training command needs a fresh
output directory; existing completed checkpoints are protected.

## Adaptivity lecture and alternative training

The [lecture review and measured training experiments](docs/adaptivity_training.md)
cover `docs/raw/week4_adaptivity.pptx`, residual final refinement and full-model-first
curriculum training. Curriculum produces measured monotonic exit accuracy, at a
cost to overall accuracy; the retained joint model is still stronger in absolute
accuracy. Neither confidence nor a validation constraint guarantees an ordering
on every unseen subject.

```bash
python train.py --model-size tiny_25k --loss-weights 0 0 1 --checkpoint-metric final-accuracy --output-dir runs/curriculum_backbone_reproduce --device cpu --threads 2
python fit_early_heads.py --checkpoint runs/curriculum_backbone_reproduce/best.pt --output-dir runs/curriculum_reproduce --device cpu --threads 2
python scripts/evaluate_run.py --run-dir runs/curriculum_reproduce --device cpu --threads 2
```

For residual refinement of an existing completed checkpoint, use
`python refine_final.py --checkpoint runs/tiny_25k/best.pt --output-dir runs/refinement_reproduce`.
Earlier decisions start unchanged. Checkpoint selection rejects final validation
accuracy regressions; zero correction is an explicit candidate, not an improvement.
New inference checkpoints record the final-head mode so evaluation and real
adaptive routing use the same logits. Original checkpoint inference remains compatible.

## Scheduled auxiliary-loss experiment

The [scheduled auxiliary-loss record](docs/scheduled_auxiliary_training.md)
reports the completed experiment: unfreeze the full-first network, ramp early
losses from zero to 0.20 / 0.30 over 30 epochs, and protect ordered validation
accuracy and source final accuracy during checkpoint selection. Early test
accuracy improved; final test accuracy lost one correct window. Old runs remain
unchanged.

```bash
python train_scheduled.py --checkpoint runs/tiny_25k_curriculum/best.pt --output-dir runs/tiny_25k_scheduled_reproduce --device cpu --threads 2
python scripts/evaluate_run.py --run-dir runs/tiny_25k_scheduled_reproduce --device cpu --threads 2
```

Exact scheduled weights, candidate eligibility, selection flags and per-exit
metrics are saved in the training histories. Validation always uses fixed target
weights so its CE remains comparable across epochs. The selected model is
recalibrated before validation threshold sweeps and actual adaptive test inference.
