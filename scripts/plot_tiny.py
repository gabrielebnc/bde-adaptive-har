"""Plot the current Tiny run and its actual fine-tuning outcome."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tiny_25k"))
    parser.add_argument("--finetuned-run-dir", type=Path)
    parser.add_argument("--base-label", default="Joint training")
    parser.add_argument("--second-label", default="Final-stage fine-tuning")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/tiny_25k"))
    args = parser.parse_args()
    if args.second_label == args.base_label:
        parser.error("The phase labels must differ")
    roots = [(args.base_label, args.run_dir),
             (args.second_label, args.finetuned_run_dir or args.run_dir.with_name(args.run_dir.name + "_finetuned"))]
    runs, rows = [], []
    for label, run in roots:
        if not (run / "evaluation_test/metrics.json").exists():
            continue
        summary = json.loads((run / "model_summary.json").read_text())
        training = json.loads((run / "training_summary.json").read_text())
        if training["smoke_run"]:
            raise ValueError("Performance plots require full-data runs")
        runs.append({"label": label, "run_dir": str(run), "model_summary": summary,
                     "training_summary": training,
                     "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()})
        for split in ("validation", "test"):
            report = json.loads((run / f"evaluation_{split}/metrics.json").read_text())
            assert report["checkpoint_epoch"] == training["best_epoch"]
            for exit_name, values in report["exits"].items():
                rows.append({"model": label, "run_dir": str(run), "split": split, "exit": exit_name,
                             "accuracy": values["accuracy"], "macro_f1": values["macro_f1"],
                             "parameters": values["parameters_up_to_exit"], "macs": values["macs_per_window"]})
    if not runs:
        raise ValueError("No completed evaluations found in the requested run directories")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "tiny_plot_data.json").write_text(json.dumps({"runs": runs, "metrics": rows}, indent=2) + "\n")
    with (output / "tiny_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    source_retained = any(r["label"] != args.base_label and "fine_tuning" in r["training_summary"]
                          and r["training_summary"]["best_epoch"] == 0
                          for r in runs)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for split in ("test", "validation"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
        styles = {args.base_label: ("#4177b5", "--", "o"),
                  args.second_label: ("#cc6335", "-", "s")}
        for ax, metric in zip(axes, ("accuracy", "macro_f1")):
            for run in runs:
                label = run["label"]
                selected = [r for r in rows if r["model"] == label and r["split"] == split]
                values = [100 * r[metric] for r in selected]
                color, linestyle, marker = styles[label]
                display_label = ("Fine-tuning: source retained" if source_retained and label != args.base_label
                                 else label)
                ax.plot([1, 2, 3], values, color=color, linestyle=linestyle, marker=marker, label=display_label)
                if label in styles:
                    for x, value in zip([1, 2, 3], values):
                        exit_name = ("exit1", "exit2", "final")[x - 1]
                        other_label = args.base_label if label != args.base_label else args.second_label
                        other = next((100 * r[metric] for r in rows if r["model"] == other_label
                                      and r["split"] == split and r["exit"] == exit_name), None)
                        same = other is not None and abs(value - other) < 1e-9
                        if label != args.base_label and x != 3 and same:
                            continue
                        caption, offset = f"{value:.2f}", -17
                        if x == 3 or (other is not None and not same):
                            if source_retained and label == args.base_label:
                                continue
                            offset = 12 if other is None or value >= other else -20
                            phase = "Retained" if source_retained else ("Updated" if label != args.base_label else ("Joint" if args.base_label == "Joint training" else "Source"))
                            caption = f"{phase} {value:.2f}"
                        ax.annotate(caption, (x, value), xytext=(0, offset),
                                    textcoords="offset points", ha="center", color=color, fontsize=10)
            ax.set_xticks([1, 2, 3], ["Exit 1", "Exit 2", "Final exit"])
            ax.set_xlim(.8, 3.2)
            # Keep every actual metric visible if the smaller heads deteriorate.
            visible = [100 * r[metric] for r in rows if r["split"] == split]
            ax.set_ylim(min(88, np.floor(min(visible)) - 1), max(95, np.ceil(max(visible)) + 1))
            ax.set_ylabel(f"{split.capitalize()} {'accuracy' if metric == 'accuracy' else 'macro F1'} (%)")
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .9), ncol=3, frameon=False, fontsize=9)
        total = runs[0]["model_summary"]["total_parameters"]
        fig.suptitle(f"Current Tiny | {total:,} parameters | exits at 30% / 70% / 100%", fontsize=14, fontweight="bold")
        seed = runs[0]["training_summary"].get("seed", 42)
        curriculum = any("curriculum" in r["training_summary"] for r in runs)
        scheduled = any("schedule" in r["training_summary"] for r in runs)
        note = (f"Fine-tuning retained the original checkpoint (no validation-loss improvement). Single seed ({seed})."
                if source_retained else (f"Scheduled auxiliary losses; all layers trainable. Checkpoints selected on validation only. Seed {seed}."
                if scheduled else (f"Curriculum: full classifier first, then early heads on frozen backbone. Validation selection only. Seed {seed}."
                if curriculum else (f"Actual completed runs. Earlier exits frozen in second phase; selection uses validation only. Single seed ({seed})."
                if len(runs) > 1 else f"Completed joint training; checkpoint selected on weighted validation loss. Single seed ({seed})."))))
        fig.text(.5, .025, note,
                 ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, .06, 1, .81))
        for extension in ("png", "svg"):
            fig.savefig(output / f"tiny_{split}_accuracy.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    summary = runs[0]["model_summary"]
    params = [e["parameters_up_to_exit"] for e in summary["exits"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(["Exit 1", "Exit 2", "Full model"], params, color=["#e4a27f", "#d27c50", "#b55220"])
    ax.invert_yaxis()
    ax.set_xlim(0, params[-1] * 1.32)
    for bar, count in zip(bars, params):
        ax.text(count + params[-1] * .025, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({100 * count / params[-1]:.2f}%)", va="center")
    ax.set_xlabel("Cumulative parameters, including earlier classifiers")
    ax.set_title("New Tiny: requested 30% / 70% / 100% weight allocation")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"tiny_parameter_budget.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {len(runs)} completed current-Tiny phases and their actual metrics under {output}")


if __name__ == "__main__":
    main()
