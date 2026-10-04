"""Gradually introduce auxiliary losses while updating the full network."""
import csv
import math
import time
from pathlib import Path

import torch

from .metrics import EXIT_NAMES, collect_logits, fit_temperatures
from .training import run_epoch
from .utils import write_json


def scheduled_weights(epoch, ramp_epochs, target):
    """Epoch 1 is final-only; the ramp's last epoch reaches the target weights."""
    if epoch < 1 or ramp_epochs < 2:
        raise ValueError("Positive epoch and at least two ramp epochs required")
    if len(target) != 3 or any(not math.isfinite(w) or w < 0 for w in target) or abs(sum(target) - 1) > 1e-6:
        raise ValueError("Three finite nonnegative target weights summing to one required")
    fraction = min((epoch - 1) / (ramp_epochs - 1), 1.)
    first, second = fraction * target[0], fraction * target[1]
    return first, second, 1. - first - second


def candidate_score(validation, baseline_final, weights):
    """Protect the baseline final accuracy and require an ordered validation curve."""
    first, second, final = [validation[f"{name}_accuracy"] for name in EXIT_NAMES]
    if not first <= second <= final or final < baseline_final:
        return None
    weighted_accuracy = sum(w * a for w, a in zip(weights, (first, second, final)))
    return final, weighted_accuracy, -validation["loss"]


def train_scheduled(model, train_loader, validation_loader, device, output_dir, source, config):
    output = Path(output_dir)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a fresh scheduled-training directory")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    target = config["loss_weights"]
    scheduled_weights(1, config["ramp_epochs"], target)
    baseline = run_epoch(model, validation_loader, device, target,
                         max_batches=config.get("max_val_batches"))
    floor = baseline["final_accuracy"]
    best_score = candidate_score(baseline, floor, target)
    if best_score is None:
        raise ValueError("Source must have ordered validation exit accuracies")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = ("Ordered validation accuracies and final >= source final; maximize final accuracy, "
                 "then weighted accuracy across exits, then minimize fixed-weight CE")
    checkpoint = {**source, "epoch": 0, "validation_loss": baseline["loss"],
                  "training_config": config, "optimizer_state": optimizer.state_dict(),
                  "scheduler_state": scheduler.state_dict(),
                  "schedule": {"source_checkpoint": config["source_checkpoint"],
                               "source_epoch": source["epoch"], "ramp_epochs": config["ramp_epochs"],
                               "baseline_final_accuracy": floor, "all_parameters_trainable": True,
                               "criterion": criterion, "epoch_zero": "Source weights retained if no candidate qualifies"}}
    # Avoid retaining historical phase metadata that no longer describes training.
    for key in ("curriculum", "refinement", "fine_tuning"):
        checkpoint.pop(key, None)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "data_metadata.json", source["data_metadata"])
    torch.save(checkpoint, output / "best.pt")
    history, bad_epochs, started = [], 0, time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        weights = scheduled_weights(epoch, config["ramp_epochs"], target)
        training = run_epoch(model, train_loader, device, weights, optimizer,
                             max_batches=config.get("max_train_batches"))
        validation = run_epoch(model, validation_loader, device, target,
                               max_batches=config.get("max_val_batches"))
        score = candidate_score(validation, floor, target)
        selected = score is not None and score > best_score
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"weight_{name}": weight for name, weight in zip(EXIT_NAMES, weights)},
               **{f"train_{k}": v for k, v in training.items()},
               **{f"validation_{k}": v for k, v in validation.items()},
               "candidate_eligible": score is not None, "checkpoint_selected": selected}
        history.append(row)
        scheduler.step()
        if selected:
            best_score, bad_epochs = score, 0
            checkpoint.update(epoch=epoch, validation_loss=validation["loss"], calibrated=False,
                              model_state={n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                              optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict())
            torch.save(checkpoint, output / "best.pt")
        elif epoch >= config["ramp_epochs"]:
            # Give the complete schedule a chance before starting early stopping.
            bad_epochs += 1
        print(f"Schedule {epoch:03d}: weights={weights[0]:.3f}/{weights[1]:.3f}/{weights[2]:.3f}, "
              f"val accuracy={validation['exit1_accuracy']:.4f}/{validation['exit2_accuracy']:.4f}/"
              f"{validation['final_accuracy']:.4f}, eligible={score is not None}, selected={selected}", flush=True)
        write_json(output / "history.json", history)
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if bad_epochs >= config["patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    selected_validation = run_epoch(model, validation_loader, device, target,
                                    max_batches=config.get("max_val_batches"))
    assert candidate_score(selected_validation, floor, target) == best_score
    if checkpoint["epoch"]:
        logits, labels = collect_logits(model, validation_loader, device)
        temperatures, calibration = fit_temperatures(logits, labels, config["calibration_steps"])
        checkpoint.update(temperatures=temperatures, calibration=calibration, calibrated=True)
        torch.save(checkpoint, output / "best.pt")
    plot_schedule(history, output / "history.png")
    result = {"epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
              "best_validation_loss": checkpoint["validation_loss"],
              "best_validation_accuracy": best_score[0], "selected_validation": selected_validation,
              "baseline_validation": baseline, "elapsed_seconds": time.perf_counter() - started,
              "data_source": source["data_metadata"]["source"],
              "smoke_run": config.get("max_train_batches") is not None or config.get("max_val_batches") is not None,
              "device": str(device), "calibration": checkpoint["calibration"],
              "schedule": checkpoint["schedule"], "selection_constraint_verified": True,
              "training_loss_note": "Scheduled train CE weights; validation CE always uses fixed target weights",
              "guarantee_scope": "Ordering and final accuracy floor enforced on validation, not guaranteed on unseen subjects"}
    write_json(output / "training_summary.json", result)
    return result


def plot_schedule(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train, changing weights")
    axes[0].plot(epochs, [r["validation_loss"] for r in history], label="Validation, fixed weights")
    axes[0].set_ylabel("Weighted CE (different objectives)")
    for name in EXIT_NAMES:
        axes[1].plot(epochs, [r[f"validation_{name}_accuracy"] for r in history], label=name)
        axes[2].plot(epochs, [r[f"weight_{name}"] for r in history], label=name)
    axes[1].set_ylabel("Validation accuracy")
    axes[2].set_ylabel("Training loss weight")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
