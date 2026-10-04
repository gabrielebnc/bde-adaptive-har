"""Train a residual correction to Exit 2 without changing the earlier exits."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import torch

from src.data import make_loader, prepare_data
from src.model import AdaptiveHAR, model_summary
from src.refinement import train_refinement
from src.utils import seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--protection-weight", type=float, default=1.)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    args = parser.parse_args()
    if min(args.lr, args.epochs, args.patience, args.threads) <= 0 or not 0 <= args.protection_weight < float("inf"):
        parser.error("Positive settings and a finite nonnegative protection weight required")
    if any(v is not None and v <= 0 for v in (args.max_train_batches, args.max_val_batches)):
        parser.error("Batch limits must be positive")
    if (Path(args.output_dir) / "best.pt").exists():
        parser.error("Choose a fresh output directory")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if (not source.get("calibrated") or source["training_config"].get("max_train_batches") is not None
            or source["training_config"].get("max_val_batches") is not None):
        parser.error("Requires a completed full-data calibrated source checkpoint")
    torch.set_num_threads(args.threads)
    metadata = source["data_metadata"]
    seed_everything(metadata["seed"])
    device = select_device(args.device)
    bundle = prepare_data(args.data_dir, seed=metadata["seed"], val_subjects=metadata["validation_subjects"],
                          normalization=metadata["normalization"])
    if bundle.metadata != metadata:
        parser.error("Data folds/normalization do not match source")
    model = AdaptiveHAR(**{**source["model_config"], "final_mode": "residual"}).to(device)
    model.load_state_dict(source["model_state"])
    model.reset_final_correction()
    config = {**source["training_config"], **vars(args), "source_checkpoint": args.checkpoint,
              "final_mode": "residual"}
    train = make_loader(bundle.train, config["batch_size"], True, config["seed"], config["num_workers"])
    validation = make_loader(bundle.validation, config["batch_size"], num_workers=config["num_workers"])
    write_json(Path(args.output_dir) / "model_summary.json", model_summary(model))
    result = train_refinement(model, train, validation, device, args.output_dir, source, config)
    print(f"Selected refinement epoch {result['best_epoch']}; epoch 0 means no learned correction")


if __name__ == "__main__":
    main()
