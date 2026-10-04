"""Joint training and validation-gap checkpoint selection for the selected model."""
import csv
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .metrics import EXIT_NAMES, collect_logits, fit_temperatures
from .utils import write_json

DEFAULT_LOSS_WEIGHTS = (0.10, 0.20, 0.70)


def exit_gap_score(validation, minimum_gap):
    """Prefer both validation gaps, then final accuracy and lower CE.

    If no epoch qualifies, keep the best final-accuracy candidate and report
    the unmet target. Test metrics never participate in checkpoint selection.
    """
    first, second, final = [validation[f"{name}_accuracy"] for name in EXIT_NAMES]
    meets_target = min(second - first, final - second) + 1e-12 >= minimum_gap
    return meets_target, final, -validation["loss"]


def joint_loss(logits, labels, weights=DEFAULT_LOSS_WEIGHTS):
    if len(weights) != 3 or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Three nonnegative loss weights summing to one are required")
    losses = [F.cross_entropy(values, labels) for values in logits]
    return sum(weight * loss for weight, loss in zip(weights, losses)), losses


def run_epoch(model, loader, device, weights=DEFAULT_LOSS_WEIGHTS, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, individual, correct, count = 0.0, [0.0] * 3, [0] * 3, 0
    with torch.set_grad_enabled(training):
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, exit_losses = joint_loss(logits, y, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss encountered")
            if training:
                loss.backward()
                optimizer.step()
            size = len(y)
            count += size
            loss_sum += loss.item() * size
            for i, (values, exit_loss) in enumerate(zip(logits, exit_losses)):
                individual[i] += exit_loss.item() * size
                correct[i] += (values.argmax(dim=1) == y).sum().item()
    if count == 0:
        raise ValueError("Epoch processed no examples")
    result = {"loss": loss_sum / count, "samples": count}
    for i, name in enumerate(EXIT_NAMES):
        result[f"{name}_loss"] = individual[i] / count
        result[f"{name}_accuracy"] = correct[i] / count
    return result


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in history]
    for split, style in [("train", "-"), ("validation", "--")]:
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], style, label=split)
        for name in EXIT_NAMES:
            axes[1].plot(epochs, [row[f"{split}_{name}_accuracy"] for row in history], style,
                         label=f"{split} {name}")
    axes[0].set_ylabel("Weighted cross-entropy")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(model, train_loader, validation_loader, device, output_dir, data_metadata, config):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best.pt").exists():
        raise FileExistsError(f"{output_dir}/best.pt exists; choose a new --output-dir")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    weights = config["loss_weights"]
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "data_metadata.json", data_metadata)
    history, best_loss, bad_epochs = [], float("inf"), 0
    best_gap_score = (False, -1., -float("inf"))
    best_validation = None
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, train_loader, device, weights, optimizer, config["max_train_batches"])
        validation = run_epoch(model, validation_loader, device, weights, max_batches=config["max_val_batches"])
        score = exit_gap_score(validation, config["min_exit_gap"])
        row = {"epoch": epoch, "lr": learning_rate,
               **{f"train_{key}": value for key, value in training.items()},
               **{f"validation_{key}": value for key, value in validation.items()},
               "validation_gap_target_met": score[0]}
        history.append(row)
        scheduler.step()
        if score > best_gap_score:
            best_loss, bad_epochs = validation["loss"], 0
            best_validation, best_gap_score = validation, score
            checkpoint = {"format_version": 1, "epoch": epoch, "validation_loss": best_loss,
                          "model_config": model.config,
                          "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                          "data_metadata": data_metadata, "training_config": config,
                          "temperatures": [1., 1., 1.], "calibrated": False}
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            bad_epochs += 1
        write_json(output_dir / "history.json", history)
        with (output_dir / "history.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        accuracies = "/".join(f"{validation[f'{name}_accuracy']:.3f}" for name in EXIT_NAMES)
        print(f"Epoch {epoch:03d}: train loss={training['loss']:.4f}, val loss={validation['loss']:.4f}, "
              f"val accuracy={accuracies}, gap target met={score[0]}", flush=True)
        if bad_epochs >= config["patience"]:
            print(f"Early stopping after {bad_epochs} epochs without improvement", flush=True)
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    # Always calibrate on full validation, including smoke runs.
    logits, labels = collect_logits(model, validation_loader, device)
    temperatures, diagnostics = fit_temperatures(logits, labels, config["calibration_steps"])
    checkpoint.update(temperatures=temperatures, calibrated=True, calibration=diagnostics)
    torch.save(checkpoint, output_dir / "best.pt")
    plot_history(history, output_dir / "history.png")
    result = {"epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
              "best_validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - started,
              "data_source": data_metadata["source"],
              "smoke_run": config["max_train_batches"] is not None or config["max_val_batches"] is not None,
              "calibration": diagnostics, "device": str(device),
              "checkpoint_selection": "Prefer both validation accuracy gaps, then maximum final accuracy, then minimum weighted CE",
              "exit_gap_target": {
                  "minimum_gap_percentage_points": 100 * config["min_exit_gap"],
                  "validation_gaps_percentage_points": [
                      100 * (best_validation["exit2_accuracy"] - best_validation["exit1_accuracy"]),
                      100 * (best_validation["final_accuracy"] - best_validation["exit2_accuracy"])],
                  "validation_target_met": best_gap_score[0],
                  "selected_validation": best_validation,
              }}
    write_json(output_dir / "training_summary.json", result)
    return result
