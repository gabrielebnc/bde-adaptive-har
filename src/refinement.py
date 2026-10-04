"""Frozen-prefix residual refinement, with validation accuracy constraints."""
import csv
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .metrics import EXIT_NAMES, collect_logits, fit_temperatures
from .training import freeze_prefix, run_epoch
from .utils import write_json


def refinement_loss(exit2, final, labels, protection_weight=1.0):
    """Protect correct training decisions' margins; this is not an inference guarantee."""
    if protection_weight < 0:
        raise ValueError("Protection weight must be nonnegative")
    base = exit2.detach()
    target_mask = F.one_hot(labels, base.shape[1]).bool()

    def margin(logits):
        true = logits.gather(1, labels[:, None]).squeeze(1)
        return true - logits.masked_fill(target_mask, -torch.inf).amax(1)

    correct = base.argmax(1) == labels
    damage = (F.relu(margin(base) - margin(final)) * correct).mean()
    ce = F.cross_entropy(final, labels)
    return ce + protection_weight * damage, ce, damage


def refinement_candidate(validation, baseline, best):
    """Rank by final accuracy, then weighted CE; reject validation regressions."""
    accuracy = validation["final_accuracy"]
    floor = max(baseline["exit1_accuracy"], baseline["exit2_accuracy"])
    return accuracy >= floor and (accuracy, -validation["loss"]) > best


def train_refinement(model, train_loader, validation_loader, device, output_dir, source, config):
    if model.config["final_mode"] != "residual":
        raise ValueError("Requires residual final logits")
    output = Path(output_dir)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a fresh refinement directory")
    freeze_prefix(model)
    frozen = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
              if n.split(".")[0] not in {"stage3", "final"}}
    weights = config["loss_weights"]
    baseline = run_epoch(model, validation_loader, device, weights,
                         max_batches=config.get("max_val_batches"))
    if baseline["final_accuracy"] < baseline["exit1_accuracy"]:
        raise ValueError("Exit 2 already trails Exit 1; this refiner cannot repair that prefix")
    best = (baseline["final_accuracy"], -baseline["loss"])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    checkpoint = {**source, "epoch": 0, "model_config": model.config,
                  "model_state": {n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                  "validation_loss": baseline["loss"], "training_config": config,
                  "temperatures": [*source["temperatures"][:2], source["temperatures"][1]],
                  "calibration": [*source["calibration"][:2], source["calibration"][1]],
                  "refinement": {"source_checkpoint": config["source_checkpoint"],
                                 "source_epoch": source["epoch"], "frozen_prefix": True,
                                 "criterion": "Final validation accuracy >= earlier exits; maximize final accuracy, then minimize weighted CE",
                                 "epoch_zero": "Zero correction: exactly Exit 2, not an improved final classifier"}}
    write_json(output / "config.json", config)
    write_json(output / "data_metadata.json", source["data_metadata"])
    torch.save(checkpoint, output / "best.pt")
    history, bad_epochs, started = [], 0, time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.eval()
        model.stage3.train()
        model.final.train()
        sums, correct, count = [0.] * 3, [0] * 3, 0
        for index, (x, y) in enumerate(train_loader):
            if config.get("max_train_batches") is not None and index >= config["max_train_batches"]:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            objective, ce, damage = refinement_loss(logits[1], logits[2], y, config["protection_weight"])
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite refinement loss")
            objective.backward()
            optimizer.step()
            count += len(y)
            for i, value in enumerate((objective, ce, damage)):
                sums[i] += value.item() * len(y)
            for i, values in enumerate(logits):
                correct[i] += (values.argmax(1) == y).sum().item()
        if not count:
            raise ValueError("Empty refinement training epoch")
        validation = run_epoch(model, validation_loader, device, weights,
                               max_batches=config.get("max_val_batches"))
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": sums[0] / count, "train_final_ce": sums[1] / count,
               "train_margin_damage": sums[2] / count, "train_samples": count,
               **{f"train_{name}_accuracy": n / count for name, n in zip(EXIT_NAMES, correct)},
               **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row)
        scheduler.step()
        accepted = refinement_candidate(validation, baseline, best)
        if accepted:
            best, bad_epochs = (validation["final_accuracy"], -validation["loss"]), 0
            checkpoint.update(epoch=epoch, validation_loss=validation["loss"],
                              model_state={n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                              optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict())
            torch.save(checkpoint, output / "best.pt")
        else:
            bad_epochs += 1
        print(f"Refine {epoch:03d}: train objective={row['train_loss']:.4f}, val CE={validation['loss']:.4f}, "
              f"val Exit 2={validation['exit2_accuracy']:.4f}, final={validation['final_accuracy']:.4f}, selected={accepted}", flush=True)
        write_json(output / "history.json", history)
        if bad_epochs >= config["patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    for name, value in frozen.items():
        if not torch.equal(model.state_dict()[name].cpu(), value):
            raise AssertionError(f"Frozen prefix state changed: {name}")
    if checkpoint["epoch"]:
        logits, labels = collect_logits(model, validation_loader, device)
        temperatures, diagnostics = fit_temperatures((logits[2],), labels, config["calibration_steps"])
        checkpoint["temperatures"] = [*source["temperatures"][:2], temperatures[0]]
        checkpoint["calibration"] = [*source["calibration"][:2], diagnostics[0]]
    checkpoint["calibrated"] = True
    torch.save(checkpoint, output / "best.pt")
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_refinement(history, output / "history.png")
    result = {"epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
              "best_validation_loss": checkpoint["validation_loss"], "best_validation_accuracy": best[0],
              "baseline_validation": baseline, "elapsed_seconds": time.perf_counter() - started,
              "data_source": source["data_metadata"]["source"],
              "smoke_run": config.get("max_train_batches") is not None or config.get("max_val_batches") is not None,
              "device": str(device), "calibration": checkpoint["calibration"],
              "refinement": checkpoint["refinement"], "frozen_state_verified": True,
              "guarantee_scope": "Validation checkpoint constraint only; no unseen-subject accuracy guarantee",
              "training_loss_note": "Train: final CE + protected-margin penalty; validation: weighted exits CE"}
    write_json(output / "training_summary.json", result)
    return result


def plot_refinement(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r["epoch"] for r in history]
    for key, label in [("train_loss", "Train CE + protection"), ("validation_loss", "Validation weighted CE")]:
        axes[0].plot(epochs, [r[key] for r in history], label=label)
    axes[0].set_ylabel("Objective (different train/validation definitions)")
    for name in EXIT_NAMES:
        axes[1].plot(epochs, [r[f"validation_{name}_accuracy"] for r in history], label=name)
    axes[1].set_ylabel("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
