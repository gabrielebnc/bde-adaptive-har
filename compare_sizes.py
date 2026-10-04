"""Compare completed size experiments and their validation policy sweeps."""
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import json

from src.evaluation import write_csv
from src.metrics import EXIT_NAMES
from src.model import MODEL_SIZES
from src.utils import write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="runs/size_comparison")
    parser.add_argument("--allow-device-differences", action="store_true",
                        help="Report historical runs trained on different devices/threads; other settings must match")
    args = parser.parse_args()
    per_exit, policies, runs = [], [], []
    reference_data, reference_config = None, None
    setting_names = ["seed", "device", "threads", "batch_size", "epochs", "patience", "lr",
                     "weight_decay", "loss_weights", "dropouts", "min_delta", "calibration_steps"]
    if args.allow_device_differences:
        setting_names = [key for key in setting_names if key not in {"device", "threads"}]
    for name in args.run_dirs:
        root = Path(name)
        data = read_json(root / "data_metadata.json")
        config = read_json(root / "config.json")
        settings = {key: config[key] for key in setting_names}
        if reference_data is None:
            reference_data, reference_config = data, settings
        if data != reference_data or settings != reference_config:
            parser.error(f"{root}: subject folds, normalization, or training settings differ")
        training = read_json(root / "training_summary.json")
        if training["smoke_run"] or data["source"] != "uci_har":
            parser.error("Size comparison requires completed real-UCI runs, without batch truncation")
        summary = read_json(root / "model_summary.json")
        size = summary.get("model_size", config["model_size"])
        runs.append({"run_dir": str(root), "model_size": size, "channels": summary.get("channels", list(MODEL_SIZES[size])),
                     "total_parameters": summary["total_parameters"],
                     "parameter_bytes_float32": summary.get("parameter_bytes_float32", 4 * summary["total_parameters"]),
                     "best_epoch": training["best_epoch"], "epochs_completed": training["epochs_completed"],
                     "best_validation_loss": training["best_validation_loss"], "device": training["device"],
                     "threads": config["threads"]})
        for split in ["validation", "test"]:
            report = read_json(root / f"evaluation_{split}" / "metrics.json")
            if report["smoke_run"] or report["checkpoint_epoch"] != training["best_epoch"]:
                parser.error(f"{root}: stale or truncated {split} evaluation")
            for exit_name in EXIT_NAMES:
                values = report["exits"][exit_name]
                per_exit.append({"model_size": size, "run_dir": str(root), "split": split, "exit": exit_name,
                                 **{key: values[key] for key in ["accuracy", "macro_f1", "parameters_up_to_exit",
                                                                "full_model_parameters", "macs_per_window"]}})
        for mode in ["normal", "low-power"]:
            report = read_json(root / f"adaptive_validation_{mode}" / "threshold_sweep.json")
            if report["split"] != "validation" or report["smoke_run"] or report["checkpoint_epoch"] != training["best_epoch"]:
                parser.error(f"{root}: stale adaptive sweep")
            # Candidate selection uses validation only: maximize accuracy, then
            # minimize MACs; macro F1 breaks any remaining ties.
            candidate = min(report["policies"], key=lambda row: (-row["accuracy"], row["average_macs"], -row["macro_f1"]))
            test_values = {"test_accuracy": None, "test_macro_f1": None, "test_average_macs": None,
                           "test_average_stages": None, "test_exit1_percent": None,
                           "test_exit2_percent": None, "test_final_percent": None}
            test_path = root / f"adaptive_test_{mode}" / "policy_metrics.json"
            if test_path.exists():
                test = read_json(test_path)
                if (test["checkpoint_epoch"] != training["best_epoch"] or test["split"] != "test"
                    or test["smoke_run"] or test["mode"] != mode
                    or test["threshold_1"] != candidate["threshold_1"]
                    or test["threshold_2"] != candidate["threshold_2"]):
                    parser.error(f"{root}: adaptive test policy does not match the validation-selected candidate")
                for key in ["accuracy", "macro_f1", "average_macs", "average_stages"]:
                    test_values[f"test_{key}"] = test[key]
                for exit_name in EXIT_NAMES:
                    test_values[f"test_{exit_name}_percent"] = test["exit_percentages"][exit_name]
            policies.append({"model_size": size, "run_dir": str(root), **candidate, **test_values})
    output = Path(args.output_dir)
    write_csv(output / "per_exit.csv", per_exit)
    write_csv(output / "validation_policy_candidates.csv", policies)
    write_json(output / "comparison.json", {"runs": runs, "per_exit": per_exit, "validation_policy_candidates": policies,
               "policy_selection": "Maximum validation accuracy, then minimum MACs, then maximum macro F1; candidates only",
               "comparison_settings": reference_config, "data_metadata": reference_data,
               "device_differences_allowed": args.allow_device_differences,
               "limitations": "One seed per architecture; validation selects checkpoints and policy candidates. MACs and parameter storage are estimates, not measured power or total RAM."})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, split in zip(axes, ["validation", "test"]):
        for size in [run["model_size"] for run in runs]:
            rows = [row for row in per_exit if row["model_size"] == size and row["split"] == split]
            ax.plot([r["macs_per_window"] / 1e6 for r in rows], [100 * r["accuracy"] for r in rows], "o-", label=size)
            for row in rows:
                ax.annotate(row["exit"], (row["macs_per_window"] / 1e6, 100 * row["accuracy"]), xytext=(4, 5), textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Executed Conv/Linear MACs per window (millions, log scale)")
        ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_computation.png", dpi=150)
    plt.close(fig)
    print(f"Saved size comparison to {output}")


if __name__ == "__main__":
    main()
