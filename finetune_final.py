"""Fine-tune Stage 3 and the final classifier without changing earlier exits."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import torch

from src.data import make_loader, prepare_data
from src.model import AdaptiveHAR, model_summary
from src.training import fine_tune_final
from src.utils import seed_everything, select_device, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    if min(args.lr, args.epochs, args.patience, args.threads) <= 0:
        parser.error("Learning rate, epochs, patience and threads must be positive")
    if (Path(args.output_dir) / "best.pt").exists():
        parser.error("Choose a fresh output directory")
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not source.get("calibrated") or source["training_config"]["max_train_batches"] is not None or source["training_config"]["max_val_batches"] is not None:
        parser.error("Requires a completed, calibrated full-data checkpoint")
    metadata = source["data_metadata"]
    torch.set_num_threads(args.threads)
    seed_everything(metadata["seed"])
    device = select_device(args.device)
    bundle = prepare_data(args.data_dir, seed=metadata["seed"], val_subjects=metadata["validation_subjects"],
                          normalization=metadata["normalization"])
    if bundle.metadata != metadata:
        parser.error("Data folds/normalization do not match source checkpoint")
    model = AdaptiveHAR(**source["model_config"]).to(device)
    model.load_state_dict(source["model_state"])
    config = {**source["training_config"], "source_checkpoint": args.checkpoint,
              "data_dir": args.data_dir, "output_dir": args.output_dir, "device": args.device,
              "threads": args.threads, "lr": args.lr, "epochs": args.epochs, "patience": args.patience}
    train = make_loader(bundle.train, config["batch_size"], True, config["seed"], config["num_workers"])
    validation = make_loader(bundle.validation, config["batch_size"], num_workers=config["num_workers"])
    write_json(Path(args.output_dir) / "model_summary.json", model_summary(model))
    result = fine_tune_final(model, train, validation, device, args.output_dir, source, config)
    print(f"Selected fine-tuning epoch {result['best_epoch']}; 0 means original checkpoint retained")


if __name__ == "__main__":
    main()
