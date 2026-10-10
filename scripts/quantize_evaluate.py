"""Post-training quantization experiment for the selected checkpoint.

Builds reduced-precision copies of the included model in memory and compares
them with the float32 reference on the same validation/test windows. The
committed checkpoint, training code and frozen policies are never modified.

Variants:
- fp16_weights: weights rounded to float16 storage, computed in float32.
- int8_static_per_tensor: int8 weights and activations, per-tensor scales.
- int8_static_per_channel: int8 activations, per-output-channel weight scales.

Each exit component (stem, stage1, exit1, ...) is quantized separately with FX
graph mode, so the unchanged confidence router in `AdaptiveHAR.adaptive_forward`
and the static fixed-depth paths run on the quantized components. Activation
ranges are calibrated on unlabeled training-fold windows only. Temperatures and
thresholds stay frozen from the float32 validation selection.
"""
import argparse
import copy
import io
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import torch
from torch import nn
from torch.ao.quantization import QConfig, QConfigMapping
from torch.ao.quantization.observer import (HistogramObserver, MinMaxObserver,
                                            PerChannelMinMaxObserver)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from src.data import make_loader, prepare_data
from src.metrics import classification_metrics, collect_logits, policy_predictions, policy_metrics
from src.model import AdaptiveHAR, model_summary
from src.pareto import sha256, static_forward
from src.utils import seed_everything, write_json

COMPONENTS = ("stem", "stage1", "exit1", "stage2", "exit2", "stage3", "final")
POLICIES = ({"id": "normal_0.98_0.8", "mode": "normal", "threshold_1": 0.98, "threshold_2": 0.80},
            {"id": "low-power_0.98", "mode": "low-power", "threshold_1": 0.98, "threshold_2": 0.0})
EXIT_LABELS = ("exit1", "exit2", "final")


def int8_qconfig(per_channel):
    activation = HistogramObserver.with_args(dtype=torch.quint8, reduce_range=False)
    weight = (PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
              if per_channel else
              MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric))
    return QConfigMapping().set_global(QConfig(activation=activation, weight=weight))


def fuse_batchnorm(model):
    """Fold eval BatchNorm into the preceding convolution (exact up to rounding)."""
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        if isinstance(module, nn.Sequential) and len(module) >= 2 and isinstance(module[0], nn.Conv1d) \
                and isinstance(module[1], nn.BatchNorm1d):
            module[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(module[0], module[1])
            module[1] = nn.Identity()
        for conv, bn in (("conv1", "bn1"), ("conv2", "bn2")):
            if hasattr(module, conv) and isinstance(getattr(module, bn, None), nn.BatchNorm1d):
                setattr(module, conv, torch.nn.utils.fusion.fuse_conv_bn_eval(getattr(module, conv), getattr(module, bn)))
                setattr(module, bn, nn.Identity())
    return model


@torch.inference_mode()
def quantize_int8(model, calibration_loader, per_channel):
    """Replace every exit component with a float-in/float-out int8 FX module."""
    quantized = copy.deepcopy(model).eval()
    mapping = int8_qconfig(per_channel)
    features = torch.zeros(1, 9, 128)
    prepared = {}
    for name in COMPONENTS:
        prepared[name] = prepare_fx(getattr(quantized, name), mapping, (features,))
        if not name.startswith(("exit", "final")):  # heads branch off; stages chain
            features = getattr(quantized, name)(features)
        setattr(quantized, name, prepared[name])
    for x, _ in calibration_loader:  # labels unused
        quantized(x)
    for name in COMPONENTS:
        setattr(quantized, name, convert_fx(prepared[name]))
    return quantized.eval()


def fp16_weights(model):
    rounded = copy.deepcopy(model).eval()
    with torch.no_grad():
        for parameter in rounded.parameters():
            parameter.copy_(parameter.half().float())
    return rounded


def serialized_bytes(model):
    buffer = io.BytesIO()
    if any("quantized" in type(m).__module__ for m in model.modules()):
        torch.save(model.state_dict(), buffer)
    elif next(model.parameters()).dtype == torch.float32 and getattr(model, "_fp16_storage", False):
        torch.save({k: v.half() for k, v in model.state_dict().items()}, buffer)
    else:
        torch.save(model.state_dict(), buffer)
    return buffer.tell()


def weight_bytes(name, model):
    """Raw weight/bias payload bytes, excluding serialization overhead."""
    parameters = sum(p.numel() for p in model.parameters())
    if name == "float32":
        return 4 * parameters
    if name == "fp16_weights":
        return 2 * parameters
    total = 0
    for module in model.modules():
        if hasattr(module, "weight") and callable(module.weight) and "quantized" in type(module).__module__:
            weight = module.weight()
            total += weight.numel()  # int8 weights
            total += 4 * (len(weight.q_per_channel_scales()) if weight.qscheme() in
                          (torch.per_channel_affine, torch.per_channel_symmetric) else 1)
            bias = module.bias() if callable(module.bias) else module.bias
            total += 4 * bias.numel() if bias is not None else 0
    return total


@torch.inference_mode()
def evaluate_variant(model, loaders, temperatures, path_flops, reference=None):
    result = {}
    for split, loader in loaders.items():
        logits, labels = collect_logits(model, loader, torch.device("cpu"))
        entry = {"exits": {}, "policies": {}}
        for label, values in zip(EXIT_LABELS, logits):
            predictions = values.argmax(1)
            metrics = classification_metrics(labels.numpy(), predictions.numpy())
            entry["exits"][label] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                                     "confusion_matrix": metrics["confusion_matrix"]}
            if reference is not None:
                ref = reference[split]["logits"][EXIT_LABELS.index(label)]
                entry["exits"][label]["agreement_with_float32"] = float((predictions == ref.argmax(1)).float().mean())
                entry["exits"][label]["max_abs_logit_error"] = float((values - ref).abs().max())
                entry["exits"][label]["mean_abs_logit_error"] = float((values - ref).abs().mean())
        for policy in POLICIES:
            predictions, exits = policy_predictions(logits, policy["threshold_1"], policy["threshold_2"],
                                                    temperatures, policy["mode"])
            report = policy_metrics(labels.numpy(), predictions.numpy(), exits)
            row = {"accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
                   "average_stages": report["average_stages"], "exit_percentages": report["exit_percentages"],
                   "average_flops": float(path_flops[exits - 1].double().mean())}
            if reference is not None:
                ref_pred, ref_exits = reference[split]["policies"][policy["id"]]
                row["prediction_agreement_with_float32"] = float((predictions == ref_pred).float().mean())
                row["exit_agreement_with_float32"] = float((exits == ref_exits).float().mean())
            entry["policies"][policy["id"]] = row
            entry.setdefault("_cache", {})[policy["id"]] = (predictions, exits)
        entry["_logits"] = logits
        result[split] = entry
    return result


@torch.inference_mode()
def check_live_execution(model, loader, temperatures, cached):
    """Verify the real compacting router matches cached-logit policy estimates."""
    mismatches = {}
    for policy in POLICIES:
        predictions, exits = [], []
        for x, _ in loader:
            out = model.adaptive_forward(x, policy["threshold_1"], policy["threshold_2"],
                                         temperatures=temperatures, mode=policy["mode"])
            predictions.append(out.predictions)
            exits.append(out.exit_indices)
        predictions, exits = torch.cat(predictions), torch.cat(exits)
        ref_pred, ref_exits = cached[policy["id"]]
        mismatches[policy["id"]] = int(((predictions != ref_pred) | (exits != ref_exits)).sum())
    return mismatches


@torch.inference_mode()
def batch1_latency(model, windows, temperatures, repeats):
    """Warmed per-window wall time (microseconds) for full depth and normal policy."""
    timings = {}
    runners = {"static_full_depth": lambda x: static_forward(model, x, 3),
               "adaptive_normal_0.98_0.8": lambda x: model.adaptive_forward(
                   x, 0.98, 0.80, temperatures=temperatures, mode="normal")}
    for name, run in runners.items():
        for x in windows[:50]:
            run(x)
        samples = []
        for _ in range(repeats):
            for x in windows:
                start = time.perf_counter_ns()
                run(x)
                samples.append((time.perf_counter_ns() - start) / 1e3)
        samples.sort()
        timings[name] = {"median_us": statistics.median(samples),
                         "p95_us": samples[int(0.95 * (len(samples) - 1))],
                         "mean_us": statistics.fmean(samples), "samples": len(samples)}
    return timings


def strip_private(results):
    return {split: {k: v for k, v in entry.items() if not k.startswith("_")} for split, entry in results.items()}


def write_report(path, summary):
    lines = ["# Post-training quantization: results", "",
             "Generated by `scripts/quantize_evaluate.py`. Temperatures and thresholds are frozen from the",
             "float32 validation selection; int8 activation ranges use unlabeled training windows only.",
             "FLOPs are unchanged by quantization; int8 executes the same Conv/Linear MACs at lower precision.", ""]
    for split in ("validation", "test"):
        lines += [f"## {split.capitalize()}", "",
                  "| Variant | Exit 1 | Exit 2 | Final | Normal 0.98/0.80 | Low-power 0.98 | Normal FLOPs/window | Final agreement |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, variant in summary["variants"].items():
            r = variant["metrics"][split]
            exits = [f"{100 * r['exits'][e]['accuracy']:.2f}%" for e in EXIT_LABELS]
            normal, low = r["policies"]["normal_0.98_0.8"], r["policies"]["low-power_0.98"]
            agreement = r["exits"]["final"].get("agreement_with_float32")
            lines.append(f"| {name} | {' | '.join(exits)} | {100 * normal['accuracy']:.2f}% | "
                         f"{100 * low['accuracy']:.2f}% | {normal['average_flops']:,.0f} | "
                         f"{'—' if agreement is None else f'{100 * agreement:.2f}%'} |")
        lines.append("")
    lines += ["## Size and measured batch-1 latency", "",
              "| Variant | Weight payload (bytes) | Serialized state_dict (bytes) | Full depth median / p95 (µs) | Normal policy median / p95 (µs) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, variant in summary["variants"].items():
        t = variant["latency"]
        full, normal = t["static_full_depth"], t["adaptive_normal_0.98_0.8"]
        lines.append(f"| {name} | {variant['weight_payload_bytes']:,} | {variant['serialized_bytes']:,} | "
                     f"{full['median_us']:.0f} / {full['p95_us']:.0f} | {normal['median_us']:.0f} / {normal['p95_us']:.0f} |")
    lines += ["", f"Latency: {summary['runtime']['platform']}, torch {summary['runtime']['torch']}, "
              f"engine `{summary['runtime']['quantized_engine']}`, {summary['runtime']['threads']} thread(s), "
              "warmed batch-1 desktop wall time. Not on-device latency or energy.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="models/adaptive_har.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/quantization")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--calibration-windows", type=int, default=2048)
    parser.add_argument("--latency-windows", type=int, default=500)
    parser.add_argument("--latency-repeats", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory {output} is not empty; choose a fresh one")
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines \
        and platform.machine() in ("arm64", "aarch64") else "fbgemm"

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint["data_metadata"]
    seed_everything(metadata["seed"])
    model = AdaptiveHAR(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    temperatures = checkpoint["temperatures"]
    bundle = prepare_data(args.data_dir, seed=metadata["seed"], val_subjects=metadata["validation_subjects"],
                          normalization=metadata["normalization"])
    for key in ["source", "channels", "class_names", "train_subjects", "validation_subjects", "test_subjects", "sizes"]:
        if bundle.metadata[key] != metadata[key]:
            raise ValueError(f"Dataset does not match saved checkpoint metadata: {key}")

    generator = torch.Generator().manual_seed(metadata["seed"])
    calibration_indices = torch.randperm(len(bundle.train), generator=generator)[:args.calibration_windows]
    calibration = make_loader(torch.utils.data.Subset(bundle.train, calibration_indices.tolist()), 128)
    loaders = {"validation": make_loader(bundle.validation, 256), "test": make_loader(bundle.test, 256)}
    latency_windows = [bundle.test[i][0][None] for i in range(min(args.latency_windows, len(bundle.test)))]

    variants = {"float32": model,
                "fp16_weights": fp16_weights(model),
                "int8_static_per_tensor": quantize_int8(fuse_batchnorm(model), calibration, per_channel=False),
                "int8_static_per_channel": quantize_int8(fuse_batchnorm(model), calibration, per_channel=True)}
    variants["fp16_weights"]._fp16_storage = True

    # Quantization keeps every Conv/Linear MAC; path costs come from the float architecture.
    path_flops = torch.tensor([2 * r["macs_per_window"] for r in model_summary(model)["exits"]])
    reference = evaluate_variant(model, loaders, temperatures, path_flops)
    reference_cache = {split: {"logits": entry["_logits"], "policies": entry["_cache"]}
                       for split, entry in reference.items()}
    summary = {"checkpoint": args.checkpoint, "checkpoint_sha256": sha256(args.checkpoint),
               "temperatures": temperatures, "policies": list(POLICIES),
               "calibration": {"source": "training fold (labels unused)", "windows": len(calibration_indices),
                               "activation_observer": "HistogramObserver, quint8",
                               "batchnorm": "folded into preceding Conv1d before quantization"},
               "macs_per_window_by_exit": [r["macs_per_window"] for r in model_summary(model)["exits"]],
               "runtime": {"platform": f"{platform.system()} {platform.machine()}", "python": platform.python_version(),
                           "torch": torch.__version__, "quantized_engine": torch.backends.quantized.engine,
                           "threads": args.threads},
               "variants": {}}
    for name, variant in variants.items():
        print(f"Evaluating {name}...", flush=True)
        metrics = reference if name == "float32" else evaluate_variant(variant, loaders, temperatures, path_flops, reference_cache)
        live = check_live_execution(variant, loaders["test"], temperatures, metrics["test"]["_cache"])
        summary["variants"][name] = {
            "metrics": strip_private(metrics),
            "live_router_mismatches_test": live,
            "weight_payload_bytes": weight_bytes(name, variant),
            "serialized_bytes": serialized_bytes(variant),
            "latency": batch1_latency(variant, latency_windows, temperatures, args.latency_repeats)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json", summary)
    write_report(output / "report.md", summary)
    torch.save(variants["int8_static_per_channel"].state_dict(), output / "adaptive_har_int8_per_channel_state.pt")
    print((output / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
