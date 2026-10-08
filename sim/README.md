# Live Adaptive HAR demonstration

One desktop window, three vertically stacked graphs, and the existing selected
checkpoint. No retraining, alternative architectures, or cached predictions.

From the repository root, after setup and UCI HAR download:

```bash
source .venv/bin/activate
python -m sim
```

The interface uses Tkinter and Matplotlib. No extra pip dependencies are needed.
Tkinter is an optional Python component: check `python -m tkinter` if the window
cannot open, and install a Python distribution with Tk support if necessary.
The native window needs a graphical desktop; headless replay is also available.

## Three real executions per window

1. **Battery-aware adaptive:** responds to the simulated battery slider.
2. **Fixed normal:** always uses the selected 0.98 / 0.80 confidence thresholds,
   unaffected by the slider.
3. **Fixed full depth:** always executes Stem and Stages 1–3 and the final head.
   Intermediate classifier heads are skipped, as in a non-adaptive deployment.

All three share the same unchanged trained weights and saved normalization, but
each independently runs inference on exactly the same input. A seeded shuffle
draws from the official test set without replacement within each pass. Once all
test windows have been seen, a new shuffled pass begins. Pass number and source
window index are shown. This is paced replay of recorded windows, not live sensor
acquisition or chronological activity transitions.

## Battery policy

The controller implements the requested thresholds:

| State / transition | Behaviour |
| --- | --- |
| Normal, until battery falls to ≤35% | Confidence-based exits at 0.98 / 0.80; Stage 3 allowed. |
| Low-power at ≤35% | Exit 1 at confidence ≥0.98; remaining windows forced to Exit 2; Stage 3 never runs. |
| Critical below 20% | Force Exit 1 for every window; Stages 2 and 3 never run. |
| Recover from critical at ≥20% | Return to low-power, unless battery is already ≥40%. |
| Recover normal at ≥40% | Resume the normal confidence policy. |

Between 35% and 40%, the previous normal/low-power state is retained. An initial
run/reset starts in normal; the current slider is then applied before its first
window. At exactly 20% the model is not critical; at exactly 35% it enters
low-power; at exactly 40% it returns to normal. The critical boundary intentionally
has no extra hysteresis, matching the requested `<20%` rule.

Changes apply at the next window boundary, never halfway through inference.
Switching modes never reloads/restarts the model or changes its trained weights;
only the controller's execution rule changes. Cumulative metrics continue across
mode switches unless Reset is clicked.
Preset buttons demonstrate shocks and recovery without carefully positioning
the slider. The battery does not auto-drain; even 0% is a simulated critical-mode
signal, not an actual power-off or a physical battery model.

These percentages are explicit design choices, not measured optimal battery
budgets. Critical mode trades reliability for a hard depth cap: a prediction is
returned even if confidence is low. Do not tune them using the replay's test labels.
For a future real device, wire its OS low-power/battery signal into this same
small controller and validate the choices on validation subjects and hardware.

## Graphs and cumulative metrics

Each graph shows per-window calculated FLOPs on the left axis and cumulative
accuracy on the right. Window markers encode exit depth using viridis colors
(dark = Exit 1, intermediate = Exit 2, light = Final). Amber/red shading marks
low-power/critical windows in the first graph. Metrics accumulate from the last
Reset, not just the visible history:

- Accuracy, macro F1 over all six classes, and last-50-window accuracy.
- Latest prediction, confidence, correctness and used exit.
- Total/average FLOPs, savings relative to paired full-depth inference, and
  percentages of windows using each exit.
- Mean and p95 inference time after warmup, plus processed-window count.

The three policies execute sequentially on one worker, rotating execution order
to reduce systematic timing-order effects. GUI work stays on the main thread.
The pace slider is a maximum requested window rate; slower inference or plotting
can reduce throughput. It is not a classification deadline guarantee.

Pause lets any in-flight window finish; Reset clears totals, mode state and
history and restarts the same seeded shuffle. Export pauses and saves
`results.json`, per-policy `windows.csv`, and `live_comparison.png` to a unique,
git-ignored `runs/sim/` directory. JSON also separates battery-aware metrics by
normal/low-power/critical state, so a cumulative average does not conceal the
shock's effects. Early macro F1 is noisy when not all six classes have appeared.

## Repeatable headless demonstration

```bash
python -m sim --headless --steps 120 \
  --battery-script '0:80,30:30,60:10,90:40' \
  --output-dir runs/sim/shock_demo_01
```

Battery-script indices are zero-based: the example processes 30 windows per
state. Headless mode runs as fast as possible rather than pacing. Output must be
a new directory to avoid overwriting previous results.

Options include `--seed`, `--device` (CPU default), `--threads`, `--battery`,
`--history`, `--interval-ms`, `--low-at`, `--recover-at`, `--critical-below`,
`--checkpoint`, and `--data-dir`. Defaults resolve the included checkpoint and
data directory relative to the project, not the shell's working directory.

## Interpretation limits

FLOPs = twice executed Conv1d/Linear MACs, counted from real executed tensor
shapes. Counts exclude bias, BatchNorm, activations, residual additions, pooling,
softmax, routing, and data movement. Fixed-full cost is 156,084 FLOPs; cumulative
adaptive paths cost 81,456 / 135,748 / 156,216 because they evaluate visited heads.
Inference timing includes routing and counting-hook overhead; it excludes data
loading, host-to-device transfer, display and pacing. Accelerators are synchronized.
Warmup is not included in metrics. These are desktop measurements, not smartwatch
latency, energy use, or battery-life savings.

No policy can execute this network for less than its 81,456-FLOP first path.
Even critical mode therefore cannot meet a 50% reduction from the 156,084-FLOP
fixed full-depth reference at the same window rate. Battery percentage is not a
linear compute-budget constraint, and this tool does not claim otherwise.

The selected checkpoint's results are exploratory and single-seed, with prior
inspection of test subjects. A short random replay or repeated passes are not
a fresh benchmark. Ground-truth labels are used only after inference for scoring,
not for choosing the next exit, thresholds or battery mode. Critical mode was not
selected as a new accuracy-optimal policy; it is the requested fixed Exit-1 cap.

Implementation references: [Tkinter](https://docs.python.org/3/library/tkinter.html)
and [Matplotlib embedding in Tk](https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_tk_sgskip.html).

## Verification

```bash
python -m unittest tests.test_sim -v
python -m unittest discover -s tests -v
```

Tests verify battery boundaries/hysteresis, truly skipped stages/heads, exact
operation counts, independent paired execution, deterministic sampling, reset,
warmup exclusion, mode switching without restart/weight changes, cumulative
metrics, safe exports and the headless CLI.

During development, headless replay and rendering were verified on the real
checkpoint/data. The native-window smoke check aborted inside macOS application
registration during Tk display initialization in the execution environment,
before creating any widgets. Interactive rendering/control behaviour therefore
still needs a check from a normal desktop terminal; importing Tk alone is not
proof that display access is available.
