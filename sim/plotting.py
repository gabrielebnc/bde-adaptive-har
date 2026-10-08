"""Three vertically stacked comparison graphs, reusable without a display."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .core import POLICIES, TITLES

COLORS = {"battery": "#157a6e", "normal": "#2678ba", "full": "#8850b1"}
MODE_COLORS = {"normal": "#157a6e", "low-power": "#c58217", "critical": "#c44d46"}


class SimulationFigure:
    def __init__(self, history=150):
        self.history = history
        self.figure = Figure(figsize=(12, 9), dpi=100, facecolor="white")
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.accuracy_axes = [ax.twinx() for ax in self.axes]
        self.figure.subplots_adjust(left=.075, right=.925, top=.84, bottom=.09, hspace=.90)
        self.figure.legend(handles=[
            Line2D([], [], color="#384a59", drawstyle="steps-mid", label="Window compute (left axis)"),
            Line2D([], [], color="#d28010", linestyle="--", label="Cumulative accuracy (right axis)"),
        ], loc="upper center", ncol=2, frameon=False, fontsize=9)
        self.figure.text(.075, .018, "Random test-window replay, not live sensing. FLOPs: Conv/Linear only, "
                         "not energy. Timing excludes display/pacing. Warmup excluded.", fontsize=8, color="#536375")

    def update(self, simulator):
        summaries = simulator.snapshot()["policies"]
        records = simulator.records[-self.history:]
        for key, ax, accuracy_ax in zip(POLICIES, self.axes, self.accuracy_axes):
            ax.clear()
            accuracy_ax.clear()
            color = COLORS[key]
            stats = summaries[key]
            mode = f" · {simulator.controller.mode.upper()}" if key == "battery" else ""
            title = f"{TITLES[key]}{mode}"
            if records:
                latest = records[-1]
                result = latest["outputs"][key]
                correctness = "correct" if result["prediction"] == latest["label"] else "WRONG"
                title += (f"\nPrediction: {simulator.class_names[result['prediction']]} · {correctness} · "
                          f"confidence {100 * result['confidence']:.1f}% · Exit {result['exit']}")
            else:
                title += "\nWaiting for first window"
            ax.set_title(title, loc="left", color=color, fontsize=10, pad=25)
            xs = [r["step"] for r in records]
            flops = [r["outputs"][key]["flops"] / 1000 for r in records]
            ax.step(xs, flops, where="mid", color=color, linewidth=1.35)
            if records:
                ax.scatter(xs, flops, s=16, c=[r["outputs"][key]["exit"] for r in records],
                           cmap="viridis", vmin=1, vmax=3, zorder=3)
                accuracy_ax.plot(xs, [100 * r["cumulative_accuracy"][key] for r in records],
                                 color="#d28010", linestyle="--", linewidth=1.25)
                if key == "battery":
                    # Shade the mode of each observed window, not future slider values.
                    for r in records:
                        if r["mode"] != "normal":
                            ax.axvspan(r["step"] - .5, r["step"] + .5,
                                       color=MODE_COLORS[r["mode"]], alpha=.10, linewidth=0)
                latest = records[-1]
                ax.set_xlim(max(.5, latest["step"] - self.history + .5),
                            max(10.5, latest["step"] + .5))
            else:
                ax.set_xlim(.5, 10.5)
            exits = "/".join(f"{v:.0f}" for v in stats["exit_percentages"])
            metrics = (f"n={stats['samples']}   Acc {100 * stats['accuracy']:.2f}%   "
                       f"F1 {100 * stats['macro_f1']:.2f}%   Last-50 acc {100 * stats['rolling_accuracy']:.1f}%   "
                       f"Exits 1/2/3: {exits}%\n"
                       f"Total {stats['total_flops'] / 1e6:.2f} MFLOPs   "
                       f"Avg {stats['average_flops'] / 1000:.1f}k/window   "
                       f"Saved {stats['compute_saving_percent']:.1f}% vs full   "
                       f"Inference {stats['latency_mean_ms']:.2f} ms (p95 {stats['latency_p95_ms']:.2f})")
            ax.text(0., 1.015, metrics, transform=ax.transAxes, fontsize=8, color="#384a59", va="bottom")
            ax.set(ylim=(70, 166), ylabel="kFLOPs/window")
            ax.set_yticks([81.456, 135.748, 156.216], ["81.5", "135.7", "156.2"])
            accuracy_ax.set(ylim=(0, 105), ylabel="Accuracy (%)")
            accuracy_ax.yaxis.tick_right()
            accuracy_ax.yaxis.set_label_position("right")
            accuracy_ax.set_yticks([0, 50, 100])
            accuracy_ax.tick_params(axis="y", colors="#b66e09", labelsize=8)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.15)
            for side in ("top",):
                ax.spines[side].set_visible(False)
                accuracy_ax.spines[side].set_visible(False)
        self.axes[-1].set_xlabel("Processed window number (same windows for all three policies)")

    def save(self, path):
        self.figure.savefig(path, dpi=160, facecolor="white")
