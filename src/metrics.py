"""Per-exit metrics, validation temperature scaling, and cached policy sweeps."""

import itertools

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.nn import functional as F

from .data import CLASS_NAMES

EXIT_NAMES = ("exit1", "exit2", "final")


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(6), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(n),
            }
            for name, p, r, f, n in zip(CLASS_NAMES, precision, recall, f1, support)
        },
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=np.arange(6)
        ).tolist(),
        "confusion_matrix_axes": "rows=true labels; columns=predictions; class order=0..5",
    }


@torch.inference_mode()
def collect_logits(model, loader, device):
    model.eval()
    outputs, labels = [[], [], []], []
    for x, y in loader:
        for bucket, logits in zip(outputs, model(x.to(device))):
            bucket.append(logits.cpu())
        labels.append(y)
    if not labels:
        raise ValueError("Cannot evaluate an empty dataset")
    return tuple(torch.cat(bucket) for bucket in outputs), torch.cat(labels)


def fit_temperatures(logits, labels, max_steps=100):
    """Fit one positive scalar per exit by validation NLL, entirely on CPU.

    Include T=1 as a fallback so numerical optimization cannot worsen NLL.
    """
    temperatures, diagnostics = [], []
    # collect_logits uses inference_mode; clone outside that context to allow
    # autograd to save the constants while differentiating the temperature.
    labels = labels.detach().cpu().clone()
    for values in logits:
        values = values.detach().cpu().clone()
        log_t = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_t], lr=0.1, max_iter=max_steps, line_search_fn="strong_wolfe"
        )
        before = F.cross_entropy(values, labels).item()

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(values / log_t.clamp(-4, 4).exp(), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(log_t.detach().clamp(-4, 4).exp())
        after = F.cross_entropy(values / temperature, labels).item()
        if not np.isfinite(after) or after > before:
            temperature, after = 1.0, before
        temperatures.append(temperature)
        diagnostics.append(
            {
                "temperature": temperature,
                "validation_nll_before": before,
                "validation_nll_after": after,
            }
        )
    return temperatures, diagnostics


def policy_predictions(logits, threshold_1, threshold_2, temperatures, mode="normal"):
    if mode not in {"normal", "low-power"}:
        raise ValueError("Invalid mode")
    if not 0 <= threshold_1 <= 1 or not 0 <= threshold_2 <= 1:
        raise ValueError("Thresholds must lie in [0, 1]")
    probabilities = [
        F.softmax(values / temperature, dim=1)
        for values, temperature in zip(logits, temperatures)
    ]
    confidence = torch.stack([p.amax(dim=1) for p in probabilities])
    predictions = torch.stack([p.argmax(dim=1) for p in probabilities])
    exits = torch.where(
        confidence[0] >= threshold_1,
        1,
        torch.where((confidence[1] >= threshold_2) | (mode == "low-power"), 2, 3),
    )
    selected = predictions.gather(0, (exits - 1).unsqueeze(0)).squeeze(0)
    return selected, exits


def policy_metrics(labels, predictions, exits):
    result = classification_metrics(labels, predictions)
    result["exit_percentages"] = {
        name: float((exits == i).float().mean() * 100)
        for i, name in enumerate(EXIT_NAMES, 1)
    }
    result["average_stages"] = float(exits.float().mean())
    return result


def threshold_sweep(
    logits, labels, thresholds_1, thresholds_2, temperatures, mode="normal"
):
    rows = []
    # threshold_2 has no effect in low-power mode, which always returns exit 2.
    combinations = itertools.product(
        thresholds_1, thresholds_2 if mode == "normal" else [1.0]
    )
    for threshold_1, threshold_2 in combinations:
        predictions, exits = policy_predictions(
            logits, threshold_1, threshold_2, temperatures, mode
        )
        report = policy_metrics(labels, predictions, exits)
        rows.append(
            {
                "mode": mode,
                "threshold_1": threshold_1,
                "threshold_2": threshold_2 if mode == "normal" else None,
                "accuracy": report["accuracy"],
                "macro_f1": report["macro_f1"],
                "exit1_percent": report["exit_percentages"]["exit1"],
                "exit2_percent": report["exit_percentages"]["exit2"],
                "final_percent": report["exit_percentages"]["final"],
                "average_stages": report["average_stages"],
            }
        )
    if not rows:
        raise ValueError("Threshold grid is empty")
    return rows
