"""Accuracy/compute frontiers and execution-based Conv/Linear accounting."""
import hashlib
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

METHOD = "conv-linear-2flops-v1"
COMPUTE_NOTE = (
    "FLOPs = 2 * Conv1d/Linear MACs, counted from executed tensor shapes. "
    "Excludes bias, BatchNorm, ReLU, residual addition, pooling, softmax and routing. "
    "Static paths skip unused heads; adaptive paths pay for every visited head. "
    "These are consistently calculated operation counts, not latency or energy."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frontier_indices(rows):
    """Minimize FLOPs, maximize accuracy; retain ties, reject invalid metrics."""
    for row in rows:
        if not math.isfinite(row["average_flops"]) or row["average_flops"] <= 0:
            raise ValueError("FLOPs must be finite and positive")
        if not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1:
            raise ValueError("Accuracy must be a finite fraction in [0, 1]")
    return [i for i, point in enumerate(rows) if not any(
        other["average_flops"] <= point["average_flops"]
        and other["accuracy"] >= point["accuracy"]
        and (other["average_flops"] < point["average_flops"]
             or other["accuracy"] > point["accuracy"])
        for other in rows)]


def mark_frontier(rows):
    indices = set(frontier_indices(rows))
    return [{**row, "pareto": i in indices} for i, row in enumerate(rows)]


def select_policies(rows):
    """Keep each mode's validation frontier; equal coordinates use first policy."""
    selected = []
    for mode in ("normal", "low-power"):
        candidates = [row for row in rows if row["mode"] == mode]
        seen = set()
        for i in frontier_indices(candidates):
            row = candidates[i]
            coordinates = (row["average_flops"], row["accuracy"])
            if coordinates not in seen:
                selected.append({key: row[key] for key in
                                 ("id", "kind", "mode", "threshold_1", "threshold_2", "depth")})
                seen.add(coordinates)
    return selected


@contextmanager
def count_macs(model):
    """Count actual executed batches, including adaptive remaining-batch sizes."""
    counts = {"macs": 0}
    handles = []

    def record(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            counts["macs"] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        else:
            counts["macs"] += output.numel() * module.in_features

    try:
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                handles.append(module.register_forward_hook(record))
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def static_forward(model, x, depth):
    """A fixed-depth model sharing checkpoint weights, with no confidence router."""
    if depth not in (1, 2, 3):
        raise ValueError("Static depth must be 1, 2 or 3")
    x = model.stage1(model.stem(x))
    if depth == 1:
        return model.exit1(x)
    x = model.stage2(x)
    if depth == 2:
        return model.exit2(x)
    return model.final(model.stage3(x))


def static_policies():
    return [{"id": f"static_exit_{depth}", "kind": "static", "mode": "static",
             "threshold_1": None, "threshold_2": None, "depth": depth}
            for depth in (1, 2, 3)]


@torch.inference_mode()
def evaluate_policy(model, loader, device, policy, temperatures):
    from .metrics import EXIT_NAMES, policy_metrics

    labels, predictions, exits = [], [], []
    with count_macs(model) as counts:
        for x, y in loader:
            x = x.to(device)
            if policy["kind"] == "static":
                prediction = static_forward(model, x, policy["depth"]).argmax(1)
                exit_index = torch.full((len(x),), policy["depth"], dtype=torch.long)
            else:
                result = model.adaptive_forward(
                    x, policy["threshold_1"], policy["threshold_2"] or 0.,
                    temperatures=temperatures, mode=policy["mode"])
                prediction, exit_index = result.predictions, result.exit_indices.cpu()
            labels.append(y)
            predictions.append(prediction.cpu())
            exits.append(exit_index)
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset")
    labels, predictions, exits = map(torch.cat, (labels, predictions, exits))
    report = policy_metrics(labels.numpy(), predictions.numpy(), exits)
    row = {**policy, "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
           "samples": len(labels), "average_stages": report["average_stages"],
           "average_macs": counts["macs"] / len(labels),
           "average_flops": 2 * counts["macs"] / len(labels)}
    row.update({f"{name}_percent": report["exit_percentages"][name] for name in EXIT_NAMES})
    return row, report


def plot_frontier(rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = [("normal", "Normal policies", "o", "#2374ab"),
              ("low-power", "Low-power policies", "s", "#ed8b23"),
              ("static", "Fixed-depth static references", "*", "#8a3e91")]
    for mode, label, marker, color in styles:
        group = [r for r in rows if r["mode"] == mode]
        if group:
            ax.scatter([r["average_flops"] / 1000 for r in group],
                       [100 * r["accuracy"] for r in group], label=label,
                       marker=marker, color=color, s=110 if mode == "static" else 34, zorder=3)
    frontier = sorted({(rows[i]["average_flops"] / 1000, 100 * rows[i]["accuracy"])
                       for i in frontier_indices(rows)})
    if frontier:
        ax.plot(*zip(*frontier), color="#263238", linewidth=1.2, linestyle="--",
                label="Non-dominated evaluated points", zorder=2)
    for row in rows:
        if row["kind"] == "static":
            ax.annotate(f"Exit {row['depth']}",
                        (row["average_flops"] / 1000, 100 * row["accuracy"]),
                        xytext=(5, -15), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Average calculated FLOPs / window (thousands)",
           ylabel="Accuracy (%)", title=title)
    ax.grid(alpha=.2)
    ax.margins(x=.1, y=.16)
    ax.legend(fontsize=8, loc="best")
    fig.text(.5, .02, "Conv/Linear only; 1 MAC = 2 FLOPs. Dashed segments guide the eye, not measured policies.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
