"""Fit auxiliary classifiers after full-depth training, preserving the full model."""
import csv
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .metrics import EXIT_NAMES, collect_logits, fit_temperatures
from .training import run_epoch
from .utils import write_json


def fit_early_heads(model, train_loader, validation_loader, device, output_dir, source, config):
    if model.config["final_mode"] != "independent":
        raise ValueError("Frozen-final curriculum requires an independent final classifier")
    output = Path(output_dir)
    if (output / "best.pt").exists():
        raise FileExistsError("Choose a fresh output directory")
    model.eval()
    for name, p in model.named_parameters():
        p.requires_grad_(name.split(".")[0] in {"exit1", "exit2"})
    frozen = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
              if n.split(".")[0] not in {"exit1", "exit2"}}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    weights = [.4, .6, 0.]
    config = {**config, "loss_weights": weights}
    baseline = run_epoch(model, validation_loader, device, weights)
    checkpoint = {**source, "epoch": 0, "training_config": config,
                  "validation_loss": baseline["loss"],
                  "curriculum": {"source_checkpoint": config["source_checkpoint"],
                                 "source_epoch": source["epoch"], "frozen_backbone_and_final": True,
                                 "criterion": "Minimum weighted early-head validation CE; no intentional weakening of heads"}}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config)
    write_json(output / "data_metadata.json", source["data_metadata"])
    torch.save(checkpoint, output / "best.pt")
    best_loss, bad_epochs, history, started = baseline["loss"], 0, [], time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.eval()
        model.exit1.train()
        model.exit2.train()
        loss_sum, correct, count = 0., [0, 0], 0
        for index, (x, y) in enumerate(train_loader):
            if config.get("max_train_batches") is not None and index >= config["max_train_batches"]:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                first = model.stage1(model.stem(x))
                second = model.stage2(first)
            logits = model.exit1(first), model.exit2(second)
            loss = .4 * F.cross_entropy(logits[0], y) + .6 * F.cross_entropy(logits[1], y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite early-head loss")
            loss.backward()
            optimizer.step()
            count += len(y)
            loss_sum += loss.item() * len(y)
            for i, values in enumerate(logits):
                correct[i] += (values.argmax(1) == y).sum().item()
        if not count:
            raise ValueError("Empty early-head epoch")
        validation = run_epoch(model, validation_loader, device, weights)
        if validation["final_accuracy"] != baseline["final_accuracy"]:
            raise AssertionError("Final predictions changed while fitting early heads")
        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / count, "train_samples": count,
               "train_exit1_accuracy": correct[0] / count, "train_exit2_accuracy": correct[1] / count,
               **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row)
        scheduler.step()
        if validation["loss"] < best_loss:
            best_loss, bad_epochs = validation["loss"], 0
            checkpoint.update(epoch=epoch, validation_loss=best_loss,
                              model_state={n: v.detach().cpu().clone() for n, v in model.state_dict().items()},
                              optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict())
            torch.save(checkpoint, output / "best.pt")
        else:
            bad_epochs += 1
        print(f"Heads {epoch:03d}: val CE={validation['loss']:.4f}, "
              f"accuracy={validation['exit1_accuracy']:.4f}/{validation['exit2_accuracy']:.4f}/{validation['final_accuracy']:.4f}", flush=True)
        write_json(output / "history.json", history)
        if bad_epochs >= config["patience"]:
            break
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    for n, value in frozen.items():
        if not torch.equal(model.state_dict()[n].cpu(), value):
            raise AssertionError(f"Frozen backbone/final changed: {n}")
    logits, labels = collect_logits(model, validation_loader, device)
    temperatures, calibration = fit_temperatures(logits[:2], labels, config["calibration_steps"])
    checkpoint.update(temperatures=[*temperatures, source["temperatures"][2]],
                      calibration=[*calibration, source["calibration"][2]], calibrated=True)
    torch.save(checkpoint, output / "best.pt")
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _plot_heads(history, output / "history.png")
    result = {"epochs_completed": len(history), "best_epoch": checkpoint["epoch"],
              "best_validation_loss": best_loss, "baseline_validation": baseline,
              "elapsed_seconds": time.perf_counter() - started,
              "data_source": source["data_metadata"]["source"],
              "smoke_run": config.get("max_train_batches") is not None,
              "device": str(device), "calibration": checkpoint["calibration"],
              "curriculum": checkpoint["curriculum"], "frozen_state_verified": True}
    write_json(output / "training_summary.json", result)
    return result


def _plot_heads(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = [r["epoch"] for r in history]
    for key in ("train_loss", "validation_loss"):
        axes[0].plot(epochs, [r[key] for r in history], label=key)
    axes[0].set_ylabel("Weighted early-head CE (0.4 / 0.6)")
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
