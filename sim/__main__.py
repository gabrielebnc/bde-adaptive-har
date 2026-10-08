"""Run a live desktop demonstration or a deterministic headless shock replay."""
import argparse
from pathlib import Path

from .core import Simulator

PROJECT = Path(__file__).resolve().parents[1]


def parse_battery_script(value):
    try:
        events = [(int(step), float(battery)) for step, battery in
                  (entry.split(":") for entry in value.split(","))]
    except ValueError as error:
        raise ValueError("Battery script format: 0:80,20:30,40:10,60:40") from error
    if (any(step < 0 or not 0 <= battery <= 100 for step, battery in events)
            or [step for step, _ in events] != sorted({step for step, _ in events})):
        raise ValueError("Battery-script indices must be unique, increasing and nonnegative; percentages in [0, 100]")
    return dict(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "models/adaptive_har.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--battery", type=float, default=80.)
    parser.add_argument("--low-at", type=float, default=35.)
    parser.add_argument("--recover-at", type=float, default=40.)
    parser.add_argument("--critical-below", type=float, default=20.)
    parser.add_argument("--interval-ms", type=int, default=250)
    parser.add_argument("--history", type=int, default=150)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--battery-script", help="Headless only: zero-based step:battery events")
    parser.add_argument("--output-dir", type=Path, help="Fresh directory for headless results")
    args = parser.parse_args()
    if not 0 <= args.battery <= 100 or args.interval_ms <= 0 or args.history <= 0 or args.steps <= 0:
        parser.error("Battery must be in [0, 100]; interval, history and steps must be positive")
    if not args.headless and (args.battery_script or args.output_dir):
        parser.error("--battery-script and --output-dir require --headless; use GUI Export otherwise")
    if args.headless and args.output_dir is None:
        parser.error("Headless mode requires a fresh --output-dir")
    if args.output_dir and args.output_dir.exists():
        parser.error("Output directory already exists; choose a new path to preserve previous results")
    try:
        events = parse_battery_script(args.battery_script) if args.battery_script else {}
        simulator = Simulator.from_checkpoint(args.checkpoint, args.data_dir, seed=args.seed,
                                             threads=args.threads, device=args.device,
                                             low_at=args.low_at, recover_at=args.recover_at,
                                             critical_below=args.critical_below)
        simulator.controller.update(args.battery)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.headless:
        simulator.warmup()
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from .plotting import SimulationFigure
        battery = args.battery
        for step in range(args.steps):
            battery = events.get(step, battery)
            simulator.step(battery)
        simulator.export(args.output_dir)
        plot = SimulationFigure(args.history)
        FigureCanvasAgg(plot.figure)
        plot.update(simulator)
        plot.save(args.output_dir / "live_comparison.png")
        for key, metrics in simulator.snapshot()["policies"].items():
            print(f"{key}: n={metrics['samples']}, accuracy={100 * metrics['accuracy']:.2f}%, "
                  f"avg FLOPs={metrics['average_flops']:.1f}, exits={metrics['exit_counts']}")
        print(f"Saved replay to {args.output_dir}")
    else:
        try:
            from .gui import launch
            launch(simulator, battery=args.battery, interval_ms=args.interval_ms, history=args.history)
        except (ImportError, RuntimeError) as error:
            parser.error(f"Desktop GUI unavailable: {error}. Check python -m tkinter, or use --headless.")


if __name__ == "__main__":
    main()
