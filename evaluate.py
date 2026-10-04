"""Evaluate every exit using the checkpoint's subject split and normalization."""
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

from src.evaluation import evaluation_parser, load_evaluation, plot_confusions, write_csv
from src.metrics import EXIT_NAMES, classification_metrics, collect_logits
from src.model import model_summary
from src.utils import write_json


def main():
    parser = evaluation_parser(__doc__)
    args = parser.parse_args()
    model, loader, device, checkpoint = load_evaluation(args)
    logits, labels = collect_logits(model, loader, device)
    summary = model_summary(model)
    reports = {name: {**classification_metrics(labels.numpy(), values.argmax(dim=1).numpy()),
                      **complexity, "full_model_parameters": summary["total_parameters"]}
               for name, values, complexity in zip(EXIT_NAMES, logits, summary["exits"])}
    output_dir = Path(args.output_dir or Path(args.checkpoint).parent / f"evaluation_{args.split}")
    write_json(output_dir / "metrics.json", {"split": args.split, "checkpoint_epoch": checkpoint["epoch"],
               "data_source": checkpoint["data_metadata"]["source"],
               "smoke_run": checkpoint["training_config"]["max_train_batches"] is not None or checkpoint["training_config"]["max_val_batches"] is not None,
               "samples": len(labels), "exits": reports, "model_summary": summary})
    write_csv(output_dir / "per_class.csv", [{"exit": name, "class": activity, **scores}
               for name, report in reports.items() for activity, scores in report["per_class"].items()])
    plot_confusions(reports, output_dir / "confusion_matrices.png")
    for name, report in reports.items():
        print(f"{name}: accuracy={report['accuracy']:.4f}, macro F1={report['macro_f1']:.4f}, "
              f"parameters through exit={report['parameters_up_to_exit']:,}, MACs={report['macs_per_window']:,}")
    print(f"Detailed metrics, per-class precision/recall, and confusion matrices: {output_dir}")


if __name__ == "__main__":
    main()
