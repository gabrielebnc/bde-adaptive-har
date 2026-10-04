# Selected Adaptive HAR model

The project uses one fixed three-exit residual CNN, with its trained checkpoint
included at [models/adaptive_har.pt](../models/adaptive_har.pt).

## Checkpoint identity

- Selected epoch: 41 of 71 completed full-data epochs.
- Seed: 42; device: CPU, two threads.
- SHA-256: `3c272d8a9856e8ec1c4d73079e350e60b8c3fde36d0a2fe870be8b3bbada2896`.
- The checkpoint is the original selected file, copied without modifying weights
  or serialization. It includes model settings/state, split metadata,
  training-fold normalization, calibrated temperatures, training settings,
  and optimizer/scheduler state.
- The original metadata name `tiny_25k` identifies this fixed architecture.
  There is no model-size selector or alternative final-head implementation.

Load only trusted checkpoints. The evaluation commands use
`torch.load(..., weights_only=True)`.

## Architecture and computation

Input shape is `(B, 9, 128)`; each head returns `(B, 6)` logits. The kernel-7
stem has stride 2 and four channels. Stages use two-convolution residual blocks
with BatchNorm/ReLU, and projected skips when needed. Stages 2 and 3 downsample
by two. Heads use global average pooling, dropout and a linear classifier.

| Component | Feature shape | Blocks | Cumulative parameters | Conv/Linear MACs/window |
| --- | --- | ---: | ---: | ---: |
| Stem | `(B, 4, 64)` | — | — | — |
| Stage 1 / Exit 1 | `(B, 4, 64)` | 4 | 738 (29.69%) | 40,728 |
| Stage 2 / Exit 2 | `(B, 7, 32)` | 3 | 1,731 (69.63%) | 67,874 |
| Stage 3 / Final | `(B, 7, 16)` | 2 | 2,486 (100%) | 78,108 |

Counts include earlier exit heads. Dropout rates are 0.40 / 0.40 / 0.10 and
have no extra inference-time cost. MACs exclude pooling, BatchNorm, activations,
softmax and routing. These are not measured latency or energy savings.

## Measured recognition quality

| Exit | Validation accuracy | Test accuracy | Validation macro F1 | Test macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Exit 1 | 91.02% | 90.53% | 91.22% | 90.53% |
| Exit 2 | 93.21% | 91.31% | 93.61% | 91.47% |
| Final | 94.64% | 92.50% | 95.01% | 92.63% |

Validation accuracy gains are 2.19 and 1.43 percentage points. Test accuracy
gains are 0.78 and 1.19 points; test macro F1 gains are 0.94 and 1.16 points.
The approximately 92% final-accuracy objective is met, but the strict one-point
first test gap is not. This limitation is preserved in the record, not rounded
away.

Exact metrics and confusion matrices are recorded in
[validation results](results/validation.json) and [test results](results/test.json).
Architecture counts are in [the model summary](results/model_summary.json).

## Subject folds and normalization

Training uses 5,392 official training windows. Validation contains 1,960 windows
from subjects 17, 21, 25, 26 and 30. Test contains 2,947 windows from subjects
2, 4, 9, 10, 12, 13, 18, 20 and 24. Entire subjects stay in a single fold.

The nine channel means and population standard deviations are fitted on training
windows/time steps only. Evaluation restores those values from the checkpoint.
Inputs must be normalized in the saved channel order before direct model calls.
Different held-out subjects help explain the validation/test accuracy difference.

## Training and calibration

The defaults in [train.py](../train.py) reproduce the selected recipe:

| Setting | Value |
| --- | --- |
| Optimizer / weight decay | AdamW / 0.0001 |
| Learning rate / schedule | 0.001 / cosine decay over 100 epochs |
| Batch size | 64 |
| Maximum epochs / patience | 100 / 30 |
| Exit loss weights | 0.10 / 0.20 / 0.70 |
| Dropout | 0.40 / 0.40 / 0.10 |
| Validation gap target | 0.01 accuracy, or one percentage point |
| Seed / CPU threads | 42 / 2 |

All layers train jointly with ordinary cross-entropy. Checkpoint selection
prefers both validation gaps, then final validation accuracy, then weighted CE.
The same validation fold fits positive temperature scales and selects adaptive
thresholds. No test labels select epochs, fit temperatures or select policies.
Temperature scaling does not change per-exit predicted classes.

[The training summary](results/training_summary.json) contains exact selection
gaps, epoch count and calibration diagnostics. The checkpoint retains complete
training/data settings. Reproduction commands are in [the README](../README.md).

## Adaptive policies

Confidence is `softmax(logits / temperature).max()`. Acceptance uses `>=`.
Normal thresholds are 0.98 / 0.80; low-power uses 0.98 at Exit 1 and then stops at
Exit 2. The remaining batch is compacted after each exit.

| Mode | Test accuracy | Macro F1 | Average stages | Average estimated MACs |
| --- | ---: | ---: | ---: | ---: |
| Normal | 92.47% | 92.59% | 1.579 | 54,231 |
| Low-power | 91.35% | 91.51% | 1.448 | 52,887 |

On test, 55.21% of windows leave at Exit 1; 13.13% reach the final exit in normal
mode. Exact records are the [validation-selected policies](results/selected_policies.json),
[normal-mode test report](results/adaptive_normal.json) and
[low-power test report](results/adaptive_low_power.json).

These are exploratory, single-seed results. Test subjects were inspected across
earlier experiments, so the numbers are not a fresh independent benchmark or
a guarantee of either gap on future users. Device/library changes can affect
reproducibility. Calibration improves validation NLL, not necessarily all
unseen-subject confidence estimates.
