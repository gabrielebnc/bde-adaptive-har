 """Joint weighted-exit training, checkpoint selection, and histories."""
import csv
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .metrics import EXIT_NAMES, collect_logits, fit_temperatures
from .utils import write_json

DEFAULT_LOSS_WEIGHTS = (0.20, 0.30, 0.50)


def joint_loss(logits, labels, weights=DEFAULT_LOSS_WEIGHTS):
    if len(weights) != 3 or any(w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError("Three nonnegative loss weights summing to one are required")
    losses = [F.cross_entropy(values, labels) for values in logits]
    return sum(weight * loss for weight, loss in zip(weights, losses)), losses


def run_epoch(model, loader, device, weights=DEFAULT_LOSS_WEIGHTS, optimizer=None, max_batches=None,
              train_final_only=False):
    training = optimizer is not None
    model.train(training)
    if training and train_final_only:
        # Freeze prefix dropout and BatchNorm statistics as well as its weights.
        model.eval()
        model.stage3.train()
        model.final.train()
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


def freeze_prefix(model):
    """Train only Stage 3 and its head; keep all prefix parameters and buffers."""
    model.eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".")[0] in {"stage3", "final"})


def fine_tune_final(model, train_loader, validation_loader, device, output_dir, source, config):
    """Retain the source as epoch-zero candidate and select on validation only."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "best.pt").exists():
        raise FileExistsError("Fine-tuning output checkpoint already exists")
    freeze_prefix(model)
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if name.split(".")[0] not in {"stage3", "final"}}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    weights = config["loss_weights"]
    baseline = run_epoch(model, validation_loader, device, weights)
    best_loss, best_epoch, bad_epochs = baseline["loss"], 0, 0
    checkpoint = {**source, "epoch": 0, "validation_loss": best_loss,
                  "training_config": config, "optimizer_state": optimizer.state_dict(),
                  "scheduler_state": scheduler.state_dict(),
                  "fine_tuning": {"source_checkpoint": config["source_checkpoint"],
                                  "source_epoch": source["epoch"], "frozen_prefix": True,
                                  "criterion": "Weighted validation loss; source included as epoch-zero candidate"}}
    torch.save(checkpoint, output_dir / "best.pt")
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "data_metadata.json", source["data_metadata"])
    history, started = [], time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        training = run_epoch(model, train_loader, device, (0., 0., 1.), optimizer,
                             train_final_only=True)
        validation = run_epoch(model, validation_loader, device, weights)
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               **{f"train_{key}": value for key, value in training.items()},
               **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        scheduler.step()
        if validation["loss"] < best_loss - config["min_delta"]:
            best_loss, best_epoch, bad_epochs = validation["loss"], epoch, 0
            checkpoint = {**checkpoint, "epoch": epoch, "validation_loss": best_loss,
                          "model_state": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                          "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                          "calibrated": False}
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            bad_epochs += 1
        print(f"Fine-tune epoch {epoch:03d}: final train CE={training['loss']:.4f}, "
              f"weighted val loss={validation['loss']:.4f}, final val accuracy={validation['final_accuracy']:.4f}", flush=True)
        if bad_epochs >= config["patience"]:
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    for name, value in frozen.items():
        if not torch.equal(model.state_dict()[name].cpu(), value):
            raise AssertionError(f"Frozen prefix state changed: {name}")
    # Earlier logits and temperatures stay unchanged; recalibrate only a changed final head.
    if best_epoch:
        logits, labels = collect_logits(model, validation_loader, device)
        final_temperature, diagnostics = fit_temperatures((logits[2],), labels, config["calibration_steps"])
        checkpoint["temperatures"] = [*source["temperatures"][:2], final_temperature[0]]
        checkpoint["calibration"] = [*source["calibration"][:2], diagnostics[0]]
    checkpoint["calibrated"] = True
    torch.save(checkpoint, output_dir / "best.pt")
    write_json(output_dir / "history.json", history)
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, output_dir / "history.png", final_only=True)
    result = {"epochs_completed": len(history), "best_epoch": best_epoch,
              "best_validation_loss": best_loss, "baseline_validation": baseline,
              "elapsed_seconds": time.perf_counter() - started,
              "data_source": source["data_metadata"]["source"], "smoke_run": False,
              "device": str(device), "calibration": checkpoint["calibration"],
              "fine_tuning": checkpoint["fine_tuning"], "frozen_state_verified": True,
              "training_loss_note": "Training loss is final CE only; validation loss uses original joint weights"}
    write_json(output_dir / "training_summary.json", result)
    return result


def plot_history(history, path, final_only=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [row["epoch"] for row in history]
    for split, style in [("train", "-"), ("validation", "--")]:
        label = f"{split} ({'final exit' if split == 'train' else 'weighted exits'})" if final_only else split
        axes[0].plot(epochs, [row[f"{split}_loss"] for row in history], style, label=label)
        for name in EXIT_NAMES:
            axes[1].plot(epochs, [row[f"{split}_{name}_accuracy"] for row in history], style, label=f"{split} {name}")
    axes[0].set_ylabel("Cross-entropy" if final_only else "Weighted cross-entropy")
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
    best_accuracy = -1.
    accuracy_selection = config.get("checkpoint_metric") == "final-accuracy"
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        training = run_epoch(model, train_loader, device, weights, optimizer, config["max_train_batches"])
        validation = run_epoch(model, validation_loader, device, weights, max_batches=config["max_val_batches"])
        row = {"epoch": epoch, "lr": learning_rate,
               **{f"train_{key}": value for key, value in training.items()},
               **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        scheduler.step()
        improved = ((validation["final_accuracy"], -validation["loss"]) > (best_accuracy, -best_loss)
                    if accuracy_selection else validation["loss"] < best_loss - config["min_delta"])
        if improved:
            best_loss, bad_epochs = validation["loss"], 0
            best_accuracy = validation["final_accuracy"]
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
        print(f"Epoch {epoch:03d}: train loss={training['loss']:.4f}, val loss={validation['loss']:.4f}, val accuracy={accuracies}", flush=True)
        if bad_epochs >= config["patience"]:
            print(f"Early stopping after {bad_epochs} epochs without improvement", flush=True)
            break
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    # Always calibrate on the full held-out validation fold, including smoke runs.
    logits, labels = collect_logits(model, validation_loader, device)
    temperatures, diagnostics = fit_temperatures(logits, labels, config["calibration_steps"])
    checkpoint.update(temperatures=temperatures, calibrated=True, calibration=diagnostics)
    torch.save(checkpoint, output_dir / "best.pt")
    plot_history(history, output_dir / "history.png")
    result = {"epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
              "best_validation_loss": best_loss, "elapsed_seconds": time.perf_counter() - started,
              "data_source": data_metadata["source"],
              "smoke_run": config["max_train_batches"] is not None or config["max_val_batches"] is not None,
              "calibration": diagnostics, "device": str(device)}
    result["checkpoint_selection"] = ("Maximum final validation accuracy, then minimum weighted CE"
                                       if accuracy_selection else "Minimum weighted validation CE")
    write_json(output_dir / "training_summary.json", result)
    return result
