# Live Adaptive HAR demonstration

A desktop window compares three execution policies of the same selected
checkpoint on identical test windows. No retraining or cached predictions.

From the repository root, after setup and dataset download:

```bash
python -m sim
```

Tkinter and Matplotlib provide the interface; no extra pip dependencies are
needed. Tkinter requires a desktop and a Python distribution with Tk support.
Check `python -m tkinter` if the window cannot open, or use headless mode below.

## Three stacked graphs

1. **Battery-aware:** follows the simulated battery controller.
2. **Fixed normal:** always uses confidence thresholds 0.98 / 0.80.
3. **Fixed full depth:** executes all stages and the final classifier, skipping
   unused early heads.

Every window is classified independently by all three policies. The seeded
shuffle visits each of the 2,947 official test windows once per pass, then
starts another shuffled pass. Saved normalization and weights stay unchanged.
This is recorded-window replay, not live sensing or chronological activity.

## Battery controller

| Battery / state | Behaviour |
| --- | --- |
| Normal | Exit on confidence ≥0.98 at Exit 1 or ≥0.80 at Exit 2; otherwise use Final. |
| ≤35% | Enter low-power: confidence-based Exit 1, otherwise force Exit 2. No Stage 3. |
| <20% | Critical: force Exit 1. No Stages 2–3. |
| ≥20% after critical | Return to low-power unless already ≥40%. |
| ≥40% | Recover normal mode. |

Between 35% and 40%, retain the previous normal/low-power state. Exactly 20%
is low-power, not critical. The critical boundary has no extra hysteresis.
Reset starts in normal and applies the current slider before the next window.

Switches apply between windows: no model restart, weight changes or checkpoint
reload. Percentages are design choices, not measured energy-optimal budgets.
The battery does not auto-drain; even 0% is a simulated critical signal, not
power-off. Critical mode returns a prediction even when confidence is low.

## Controls and metrics

- Start/Pause; an in-flight window finishes before pausing.
- Battery slider plus normal, shock, critical and recovery presets.
- Requested maximum window rate; rendering/inference can reduce actual throughput.
- Reset clears metrics and restarts the same seeded shuffle.
- Export pauses and writes `results.json`, `windows.csv` and
  `live_comparison.png` under a unique, ignored `runs/sim/` directory.

Graphs show per-window FLOPs and cumulative accuracy. Marker colors indicate
Exit 1/2/Final; amber/red shading marks low-power/critical windows. Readouts
include latest prediction/confidence, accuracy, six-class macro F1, last-50
accuracy, exit percentages, total/average compute, savings versus full depth,
and mean/p95 inference time. Totals cover all windows since Reset, not just the
visible history. Exported JSON also separates battery-aware results by mode.

One worker performs real inference sequentially, rotating policy order. Warmup
runs on that worker and does not count toward metrics. GUI work stays on the
main thread.

## Headless replay and full-set checks

```bash
python -m sim --headless --steps 120 \
  --battery-script '0:80,30:30,60:10,90:40' \
  --output-dir runs/sim/shock_demo_01

python -m sim --headless --steps 2947 --battery 80 \
  --output-dir runs/sim/normal_full_pass_01
```

Battery-script indices are zero-based. Headless mode runs without pacing and
requires a fresh output directory. For full-set low-power or critical checks,
use battery 30 or 10 respectively. Short replays can favor an early exit by
chance; repeated passes do not create new independent test data.

See `python -m sim --help` for seed, device, thread, history, pacing, battery
threshold, checkpoint and data-directory options. CPU is the default; paths
resolve relative to the project.

## Measurement limits

The simulator reuses the Pareto evaluator's executed Conv/Linear counter and
full-depth path. FLOPs = 2 × MACs; exclude bias, normalization, activations,
pooling, residual additions, softmax, routing and data movement. Static full
depth costs 156,084 FLOPs; adaptive paths cost 81,456 / 135,748 / 156,216.

Warmed batch-1 timing includes routing and counting hooks, with accelerator
synchronization. It excludes loading, host-to-device transfer, display and pacing.
Desktop timings/operation counts are not smartwatch energy or battery savings.
Even forced Exit 1 cannot halve the fixed-full FLOPs at the same window rate.

The checkpoint is exploratory and single-seed, with previously inspected test
subjects. Labels score predictions after inference; they never choose an exit
or mode. Changing battery produces a mixture of policies, not one fixed expected
accuracy. See [the model record](../docs/model.md) for reference metrics.

Run `python -m unittest tests.test_sim -v` for controller, inference, sampling,
metric, export and headless-rendering checks.
