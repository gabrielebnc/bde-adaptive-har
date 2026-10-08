"""GUI-independent real inference, fair sampling, and cumulative accounting."""
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.evaluation import load_evaluation, write_csv
from src.pareto import COMPUTE_NOTE, count_macs, sha256, static_forward
from src.utils import write_json
from .policy import BatteryController

POLICIES = ("battery", "normal", "full")
TITLES = {"battery": "Battery-aware adaptive model",
          "normal": "Adaptive model · fixed normal mode",
          "full": "Fixed full-depth model"}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class Prediction:
    prediction: int
    confidence: float
    exit: int
    flops: int
    latency_ms: float


@torch.inference_mode()
def infer(model, x, temperatures, mode):
    """Each call genuinely executes its path; full depth omits early classifiers."""
    if mode not in ("normal", "low-power", "critical", "full"):
        raise ValueError("Unknown inference mode")
    if model.training or len(x) != 1:
        raise ValueError("Simulation inference requires eval mode and one window")
    with count_macs(model) as count:
        synchronize(x.device)
        started = time.perf_counter()
        if mode == "full":
            logits = static_forward(model, x, 3) / temperatures[2]
            confidence, prediction = logits.softmax(1).max(1)
            exit_index = 3
        else:
            result = model.adaptive_forward(
                x, 0. if mode == "critical" else .98, .80,
                temperatures=temperatures, mode="normal" if mode == "critical" else mode)
            prediction, confidence = result.predictions, result.confidence
            exit_index = int(result.exit_indices.item())
        synchronize(x.device)
        elapsed = (time.perf_counter() - started) * 1000
    return Prediction(int(prediction.item()), float(confidence.item()), exit_index,
                      2 * count["macs"], elapsed)


class Metrics:
    def __init__(self, classes):
        self.confusion = np.zeros((classes, classes), dtype=np.int64)
        self.exit_counts = np.zeros(3, dtype=np.int64)
        self.total_flops = 0
        self.latencies = []
        self.recent_correct = deque(maxlen=50)

    def add(self, label, prediction):
        self.confusion[label, prediction.prediction] += 1
        self.exit_counts[prediction.exit - 1] += 1
        self.total_flops += prediction.flops
        self.latencies.append(prediction.latency_ms)
        self.recent_correct.append(label == prediction.prediction)

    @property
    def samples(self):
        return int(self.confusion.sum())

    @property
    def accuracy(self):
        return float(np.trace(self.confusion) / self.samples) if self.samples else 0.

    def snapshot(self):
        n = self.samples
        true_positive = np.diag(self.confusion)
        denominators = self.confusion.sum(0) + self.confusion.sum(1)
        f1 = np.divide(2 * true_positive, denominators,
                       out=np.zeros(len(true_positive), dtype=float), where=denominators > 0)
        return {"samples": n, "accuracy": self.accuracy,
                "macro_f1": float(f1.mean()),
                "rolling_accuracy": sum(self.recent_correct) / len(self.recent_correct) if n else 0.,
                "total_flops": self.total_flops,
                "average_flops": self.total_flops / n if n else 0.,
                "exit_counts": self.exit_counts.tolist(),
                "exit_percentages": (100 * self.exit_counts / n).tolist() if n else [0., 0., 0.],
                "average_stages": float(self.exit_counts @ np.arange(1, 4) / n) if n else 0.,
                "latency_mean_ms": float(np.mean(self.latencies)) if n else 0.,
                "latency_p95_ms": float(np.percentile(self.latencies, 95)) if n else 0.,
                "confusion_matrix": self.confusion.tolist()}


class Simulator:
    def __init__(self, model, dataset, temperatures, class_names, *, seed=42,
                 controller=None, provenance=None):
        if not len(dataset):
            raise ValueError("Simulation dataset must not be empty")
        if len(temperatures) != 3 or any(not np.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError("Three positive finite temperatures are required")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.dataset, self.temperatures = dataset, list(temperatures)
        self.class_names, self.seed = list(class_names), seed
        self.controller = controller or BatteryController()
        self.provenance = provenance or {}
        self.reset()

    @classmethod
    def from_checkpoint(cls, checkpoint, data_dir, *, seed=42, threads=2, device="cpu",
                        low_at=35., recover_at=40., critical_below=20.):
        args = SimpleNamespace(checkpoint=str(checkpoint), data_dir=str(data_dir), split="test",
                               batch_size=1, num_workers=0, threads=threads, device=device)
        model, loader, _, saved = load_evaluation(args)
        if not saved.get("calibrated", False):
            raise ValueError("Use a checkpoint calibrated on validation")
        provenance = {"checkpoint": str(Path(checkpoint).resolve()),
                      "checkpoint_sha256": sha256(checkpoint),
                      "data_metadata": saved["data_metadata"], "device": str(next(model.parameters()).device),
                      "threads": threads, "torch_version": str(torch.__version__),
                      "synthetic_data": saved["data_metadata"]["source"] != "uci_har"}
        return cls(model, loader.dataset, saved["temperatures"], saved["data_metadata"]["class_names"],
                   seed=seed, controller=BatteryController(low_at, recover_at, critical_below),
                   provenance=provenance)

    def reset(self):
        self.controller.reset()
        self.rng = np.random.default_rng(self.seed)
        self.order = self.rng.permutation(len(self.dataset))
        self.cursor, self.pass_number = 0, 1
        self.records = []
        self.metrics = {key: Metrics(len(self.class_names)) for key in POLICIES}
        self.mode_metrics = {mode: Metrics(len(self.class_names))
                             for mode in ("normal", "low-power", "critical")}

    def warmup(self, repeats=2):
        x = self.dataset[0][0].unsqueeze(0).to(self.device)
        for _ in range(repeats):
            for mode in ("normal", "low-power", "critical", "full"):
                infer(self.model, x, self.temperatures, mode)

    def step(self, battery):
        # Validate the signal before consuming a sample or changing accounting.
        mode = self.controller.update(battery)
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(len(self.dataset))
            self.cursor, self.pass_number = 0, self.pass_number + 1
        index = int(self.order[self.cursor])
        x, label = self.dataset[index]
        label = int(label)
        x = x.unsqueeze(0).to(self.device)
        outputs = {}
        # Rotate execution order to reduce a systematic first/last timing advantage.
        offset = len(self.records) % len(POLICIES)
        execution_order = POLICIES[offset:] + POLICIES[:offset]
        modes = {"battery": mode, "normal": "normal", "full": "full"}
        for key in execution_order:
            outputs[key] = infer(self.model, x, self.temperatures, modes[key])
        # Commit all three observations together, only after successful inference.
        cumulative = {}
        for key in POLICIES:
            self.metrics[key].add(label, outputs[key])
            cumulative[key] = self.metrics[key].accuracy
        self.mode_metrics[mode].add(label, outputs["battery"])
        record = {"step": len(self.records) + 1, "dataset_index": index,
                  "pass": self.pass_number, "label": label, "battery": float(battery), "mode": mode,
                  "outputs": {key: asdict(value) for key, value in outputs.items()},
                  "cumulative_accuracy": cumulative}
        self.records.append(record)
        self.cursor += 1
        return record

    def snapshot(self):
        summaries = {key: value.snapshot() for key, value in self.metrics.items()}
        full_flops = summaries["full"]["total_flops"]
        for summary in summaries.values():
            summary["compute_saving_percent"] = (
                100 * (1 - summary["total_flops"] / full_flops) if full_flops else 0.)
        return {"policies": summaries,
                "battery_modes": {mode: stats.snapshot() for mode, stats in self.mode_metrics.items()}}

    def export(self, directory):
        if not self.records:
            raise ValueError("Run at least one window before exporting")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        report = {"schema_version": 1, "purpose": "Live demonstration, not a new benchmark",
                  "sampling": "Seeded shuffled test windows; no replacement within each pass; repeated passes allowed",
                  "seed": self.seed, "dataset_size": len(self.dataset), "classes": self.class_names,
                  "controller": {"low_at": self.controller.low_at, "recover_at": self.controller.recover_at,
                                 "critical_below": self.controller.critical_below,
                                 "battery_is_simulated": True},
                  "thresholds": {"first": .98, "second_normal": .80},
                  "temperatures": self.temperatures, "provenance": self.provenance,
                  "compute_note": COMPUTE_NOTE,
                  "timing_note": "Warmed sequential batch-1 inference, accelerator synchronized. Includes routing "
                                 "and counting hooks; excludes loading, host-to-device transfer, plotting and pacing.",
                  **self.snapshot(), "windows": self.records}
        write_json(directory / "results.json", report)
        flat = []
        for record in self.records:
            for key in POLICIES:
                output = record["outputs"][key]
                flat.append({k: record[k] for k in ("step", "dataset_index", "pass", "label", "battery", "mode")}
                            | {"policy": key, **output,
                               "correct": output["prediction"] == record["label"],
                               "cumulative_accuracy": record["cumulative_accuracy"][key]})
        write_csv(directory / "windows.csv", flat)
        return directory
