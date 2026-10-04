"""Sweep validation thresholds or measure a fixed adaptive policy on test subjects."""
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import torch

from src.evaluation import evaluation_parser, load_evaluation, write_csv
from src.metrics import collect_logits, policy_metrics, threshold_sweep
from src.model import model_summary
from src.utils import write_json


def parse_grid(value):
    values = [float(item) for item in value.split(",")]
    if not values or any(not 0 <= item <= 1 for item in values):
        raise ValueError("Threshold grid must contain comma-separated numbers in [0, 1]")
    return values


def plot_sweep(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    points = ax.scatter([r["average_stages"] for r in rows], [r["accuracy"] for r in rows],
                        c=[r["macro_f1"] for r in rows], cmap="viridis")
    fig.colorbar(points, ax=ax, label="Macro F1")
    ax.set_xlabel("Average stages executed")
    ax.set_ylabel("Validation accuracy")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = evaluation_parser(__doc__)
    parser.set_defaults(split="validation")
    parser.add_argument("--mode", choices=["normal", "low-power"], default="normal")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--thresholds-1", default="0,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1")
    parser.add_argument("--thresholds-2", default="0,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1")
    parser.add_argument("--threshold-1", type=float)
    parser.add_argument("--threshold-2", type=float)
    args = parser.parse_args()
    if args.sweep and args.split != "validation":
        parser.error("Select thresholds on validation subjects; test sweeps are disabled")
    if args.sweep and (args.threshold_1 is not None or args.threshold_2 is not None):
        parser.error("Use threshold grids with --sweep, or single thresholds without --sweep")
    if not args.sweep and (args.threshold_1 is None or (args.mode == "normal" and args.threshold_2 is None)):
        parser.error("Specify --sweep or explicit --threshold-1 and --threshold-2 (normal mode)")
    if any(t is not None and not 0 <= t <= 1 for t in [args.threshold_1, args.threshold_2]):
        parser.error("Thresholds must lie in [0, 1]")
    model, loader, device, checkpoint = load_evaluation(args)
    if not checkpoint.get("calibrated", False):
        parser.error("Checkpoint lacks validation confidence calibration; finish training first")
    temperatures = checkpoint["temperatures"]
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / f"adaptive_{args.split}_{args.mode}")
    metadata = {"split": args.split, "mode": args.mode, "checkpoint_epoch": checkpoint["epoch"],
                "data_source": checkpoint["data_metadata"]["source"],
                "temperatures": temperatures, "samples": len(loader.dataset),
                "smoke_run": checkpoint["training_config"]["max_train_batches"] is not None or checkpoint["training_config"]["max_val_batches"] is not None}
    if args.sweep:
        try:
            grid1, grid2 = parse_grid(args.thresholds_1), parse_grid(args.thresholds_2)
        except ValueError as error:
            parser.error(str(error))
        logits, labels = collect_logits(model, loader, device)
        rows = threshold_sweep(logits, labels, grid1, grid2, temperatures, args.mode)
        macs = [r["macs_per_window"] for r in model_summary(model)["exits"]]
        for row in rows:
            row["average_macs"] = sum(row[key] / 100 * cost for key, cost in
                                      zip(["exit1_percent", "exit2_percent", "final_percent"], macs))
        metadata["execution"] = "Offline sweep using cached all-exit logits; stages/MACs are policy estimates"
        write_csv(output_dir / "threshold_sweep.csv", rows)
        write_json(output_dir / "threshold_sweep.json", {**metadata, "policies": rows})
        plot_sweep(rows, output_dir / "threshold_sweep.png")
        print(f"Evaluated {len(rows)} validation policies. Choose thresholds from {output_dir / 'threshold_sweep.csv'}")
    else:
        labels, predictions, exits = [], [], []
        with torch.inference_mode():
            for x, y in loader:
                output = model.adaptive_forward(x.to(device), args.threshold_1,
                                               1.0 if args.threshold_2 is None else args.threshold_2,
                                               temperatures=temperatures, mode=args.mode)
                labels.append(y)
                predictions.append(output.predictions.cpu())
                exits.append(output.exit_indices.cpu())
        report = policy_metrics(torch.cat(labels).numpy(), torch.cat(predictions).numpy(), torch.cat(exits))
        macs = [r["macs_per_window"] for r in model_summary(model)["exits"]]
        report["average_macs"] = sum(report["exit_percentages"][name] / 100 * cost for name, cost in
                                    zip(["exit1", "exit2", "final"], macs))
        write_json(output_dir / "policy_metrics.json", {**metadata, "threshold_1": args.threshold_1,
                   "threshold_2": args.threshold_2 if args.mode == "normal" else None,
                   "execution": "Actual adaptive inference with remaining-batch compaction", **report})
        print(f"Accuracy={report['accuracy']:.4f}, macro F1={report['macro_f1']:.4f}, "
              f"average stages={report['average_stages']:.3f}; exit percentages={report['exit_percentages']}")
        print(f"Saved {output_dir / 'policy_metrics.json'}")


if __name__ == "__main__":
    main()
