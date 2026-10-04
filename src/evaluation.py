"""Shared checkpoint loading and artifact output for both evaluation commands."""
import csv
from pathlib import Path

import torch

from .data import make_loader, prepare_data
from .model import AdaptiveHAR
from .utils import seed_everything, select_device


def evaluation_parser(description):
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", default="models/adaptive_har.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output-dir")
    return parser


def artifact_directory(args, name):
    """Keep reports generated from the committed checkpoint under ignored runs."""
    checkpoint_path = Path(args.checkpoint)
    project = Path(__file__).resolve().parents[1]
    base = (project / "runs/adaptive_har" if checkpoint_path.resolve().parent == project / "models"
            else checkpoint_path.parent)
    return Path(args.output_dir) if args.output_dir else base / name


def load_evaluation(args):
    if args.batch_size <= 0 or args.num_workers < 0 or args.threads <= 0:
        raise ValueError("Batch size and threads must be positive; workers must be nonnegative")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = checkpoint["data_metadata"]
    seed_everything(metadata["seed"])
    device = select_device(args.device)
    model = AdaptiveHAR(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    bundle = prepare_data(args.data_dir, seed=metadata["seed"], val_subjects=metadata["validation_subjects"],
                          normalization=metadata["normalization"])
    for key in ["source", "channels", "class_names", "train_subjects", "validation_subjects", "test_subjects", "sizes"]:
        if bundle.metadata[key] != metadata[key]:
            raise ValueError(f"Dataset does not match saved checkpoint metadata: {key}")
    loader = make_loader(getattr(bundle, args.split), args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda")
    return model, loader, device, checkpoint


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusions(reports, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .data import CLASS_NAMES
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, report) in zip(axes, reports.items()):
        matrix = report["confusion_matrix"]
        ax.imshow(matrix, cmap="Blues")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=8)
        ax.set_title(name)
        ax.set_xticks(range(6), CLASS_NAMES, rotation=65, ha="right", fontsize=7)
        ax.set_yticks(range(6), CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
