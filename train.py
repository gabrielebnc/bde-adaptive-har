"""Train all three exits on official training subjects only."""
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import argparse
import math

import torch

from src.data import make_loader, prepare_data
from src.model import DEFAULT_DROPOUTS, AdaptiveHAR, model_summary
from src.training import DEFAULT_LOSS_WEIGHTS, train_model
from src.utils import seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", default="runs/adaptive_har_retrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-subjects", nargs="+", type=int)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=30)
    parser.set_defaults(checkpoint_metric="exit-gap")
    parser.add_argument("--min-exit-gap", type=float, default=0.01,
                        help="Minimum adjacent validation accuracy gain for exit-gap selection; 0.01 = 1 percentage point")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-weights", nargs=3, type=float, default=list(DEFAULT_LOSS_WEIGHTS))
    parser.add_argument("--dropouts", nargs=3, type=float, default=list(DEFAULT_DROPOUTS))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-train-batches", type=int, help="Smoke test only; truncates each training epoch")
    parser.add_argument("--max-val-batches", type=int, help="Smoke test only; truncates checkpoint validation")
    parser.add_argument("--calibration-steps", type=int, default=100)
    args = parser.parse_args()
    for name in ["batch_size", "epochs", "patience", "threads", "calibration_steps", "max_train_batches", "max_val_batches"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        parser.error("Invalid optimizer/worker configuration")
    if not math.isfinite(args.min_exit_gap) or not 0 <= args.min_exit_gap <= 0.5:
        parser.error("--min-exit-gap must be finite and between 0 and 0.5")
    if any(w < 0 for w in args.loss_weights) or abs(sum(args.loss_weights) - 1) > 1e-6:
        parser.error("Loss weights must be nonnegative and sum to one")
    if any(not 0 <= d < 1 for d in args.dropouts):
        parser.error("Dropout rates must lie in [0, 1)")
    output_dir = Path(args.output_dir)
    if (output_dir / "best.pt").exists():
        parser.error("Output checkpoint already exists; choose a new --output-dir")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    bundle = prepare_data(args.data_dir, download=args.download, val_fraction=args.val_fraction,
                          seed=args.seed, val_subjects=args.val_subjects)
    print(f"Device: {device}; split sizes: {bundle.metadata['sizes']}")
    print(f"Training class counts: {bundle.metadata['class_counts']['train']}; ordinary cross-entropy")
    train_loader = make_loader(bundle.train, args.batch_size, True, args.seed, args.num_workers, device.type == "cuda")
    validation_loader = make_loader(bundle.validation, args.batch_size, num_workers=args.num_workers,
                                   pin_memory=device.type == "cuda")
    model = AdaptiveHAR(dropouts=args.dropouts).to(device)
    summary = model_summary(model)
    write_json(output_dir / "model_summary.json", summary)
    print(f"Model parameters: {summary['total_parameters']:,}")
    if args.max_train_batches or args.max_val_batches:
        print("SMOKE RUN: truncated epochs; results are not trained-model performance")
    result = train_model(model, train_loader, validation_loader, device, output_dir, bundle.metadata, vars(args))
    print(f"Saved best checkpoint (epoch {result['best_epoch']}) to {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
