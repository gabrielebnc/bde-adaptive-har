# Post-training quantization (experiment)

Exploratory experiment on branch `experiment/quantization`. The included
`models/adaptive_har.pt`, training code, temperatures and frozen policies are
unchanged; quantized copies are built in memory.

```bash
python scripts/quantize_evaluate.py --output-dir runs/quantization
```

## Method

- **fp16_weights:** weights rounded to float16, computed in float32.
- **int8 static (per-tensor / per-channel weights):** BatchNorm folded into the
  preceding convolution, then each component (stem, stages, exit heads) is
  quantized with PyTorch FX post-training quantization (`qnnpack` on arm64).
  Activation ranges are calibrated on 2,048 unlabeled training-fold windows.
  Components keep float inputs/outputs, so the existing confidence router and
  static paths run unchanged.
- Temperatures (0.171 / 0.340 / 0.609) and thresholds (0.98 / 0.80) stay frozen
  from the float32 validation selection; nothing was re-selected on test.
  The real compacting router agrees exactly with cached-logit estimates for
  every variant.

## Results

Full tables: [report](results/quantization/report.md) and
[raw results](results/quantization/results.json).

| Test | Exit 1 | Exit 2 | Final | Normal 0.98/0.80 | Normal FLOPs/window | Weight bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| float32 | 90.53% | 91.31% | 92.50% | 92.47% | 108,462 | 9,944 |
| fp16 weights | 90.50% | 91.31% | 92.50% | 92.47% | 108,469 | 4,972 |
| int8 per-tensor | 89.11% | 91.79% | 92.30% | 92.33% | 109,820 | 2,876 |
| int8 per-channel | 88.94% | 91.69% | 92.50% | 92.47% | 110,983 | 3,332 |

## Interpretation and limits

- int8 keeps final and adaptive accuracy within about 0.2 points with roughly
  3.0–3.5× smaller weight payload. fp16 is effectively lossless at 2×.
- Exit 1 loses about 1.4–1.6 test points: its four-channel features are the
  most sensitive, and the sharp temperature (0.17) amplifies logit error. That
  widens the Exit-1 → Final gap, and fewer windows pass the 0.98 threshold
  (55.2% → 50.4% per-channel), so normal-mode FLOPs rise by up to 2.3%.
  A deployed int8 model should refit temperatures and thresholds on validation.
- Quantization does not reduce FLOP counts. On this desktop CPU, PyTorch's int8
  path is about 2× **slower** at batch 1, because per-component quantize/dequantize
  and kernel dispatch dominate a 2.5k-parameter model. Serialized `state_dict`
  size is dominated by metadata. Neither reflects a microcontroller int8 runtime;
  on-device latency/energy remain unmeasured.
- Single seed, previously inspected test subjects, as for the float model.
