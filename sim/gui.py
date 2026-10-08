"""Tk window; all inference runs on one worker, all GUI work on the main thread."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from .plotting import SimulationFigure


def launch(simulator, *, battery=80., interval_ms=250, history=150):
    import tkinter as tk
    from tkinter import messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f"Cannot open a desktop display: {error}") from error
    root.title("Adaptive HAR · Live resource-shock demonstration")
    root.geometry("1300x1050")
    root.minsize(1050, 850)
    executor = ThreadPoolExecutor(max_workers=1, initializer=simulator.warmup)
    running, future, pending_action = False, None, None
    due = 0.
    battery_value, rate = tk.DoubleVar(value=battery), tk.DoubleVar(value=1000 / interval_ms)
    status = tk.StringVar(value="Ready · Start to run real inference on shuffled test windows")
    requested = tk.StringVar()
    header = ttk.Frame(root, padding=(12, 8))
    header.pack(fill="x")
    ttk.Label(header, text="Adaptive HAR · same test window, three real execution policies",
              font=("Helvetica", 16, "bold")).pack(anchor="w")
    ttk.Label(header, text="Battery affects the first model only. Amber shading = low-power; red = forced Exit 1.").pack(anchor="w")
    controls = ttk.Frame(root, padding=(12, 3))
    controls.pack(fill="x")

    def toggle():
        nonlocal running
        running = not running
        start_button.configure(text="Pause" if running else "Start")
        status.set("Running" if running else "Paused (any in-flight window will finish)")

    def request_action(action):
        nonlocal running, pending_action
        running, pending_action = False, action
        start_button.configure(text="Start")
        status.set(f"{action.title()} requested; waiting for any in-flight window")

    start_button = ttk.Button(controls, text="Start", command=toggle)
    start_button.pack(side="left")
    ttk.Button(controls, text="Reset", command=lambda: request_action("reset")).pack(side="left", padx=5)
    ttk.Button(controls, text="Export results + plot",
               command=lambda: request_action("export")).pack(side="left", padx=5)
    ttk.Label(controls, text="Max windows/sec:").pack(side="left", padx=(20, 4))
    ttk.Scale(controls, from_=1, to=10, variable=rate, length=150).pack(side="left")
    power = ttk.Frame(root, padding=(12, 5))
    power.pack(fill="x")
    ttk.Label(power, text="Simulated battery:").pack(side="left")
    ttk.Scale(power, from_=0, to=100, variable=battery_value, length=340).pack(side="left", padx=8)
    ttk.Label(power, textvariable=requested, width=25).pack(side="left")
    for value, label in [(80, "Normal (80%)"), (30, "Shock (30%)"), (10, "Critical (10%)"), (40, "Recover (40%)")]:
        ttk.Button(power, text=label, command=lambda v=value: battery_value.set(v)).pack(side="left", padx=3)
    ttk.Label(root, text=(f"Enter low-power ≤{simulator.controller.low_at:g}% · Recover ≥"
                         f"{simulator.controller.recover_at:g}% · Force Exit 1 <"
                         f"{simulator.controller.critical_below:g}% · slider applied at the next window"),
              padding=(12, 2)).pack(anchor="w")
    ttk.Label(root, textvariable=status, padding=(12, 4), wraplength=1250).pack(fill="x")
    plot = SimulationFigure(history)
    plot.update(simulator)
    canvas = FigureCanvasTkAgg(plot.figure, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    canvas.draw()

    def tick():
        nonlocal future, due, running, pending_action
        requested.set(f"{battery_value.get():.1f}% (requested)")
        if future is not None and future.done():
            try:
                record = future.result()
                status.set(f"Window {record['step']} · Test index {record['dataset_index']} · "
                           f"Pass {record['pass']} · Truth: {simulator.class_names[record['label']]} · "
                           f"Applied battery {record['battery']:.1f}% → {record['mode']}" +
                           (" · Paused" if not running else ""))
            except Exception as error:
                running = False
                start_button.configure(text="Start")
                messagebox.showerror("Inference failed", str(error))
            future = None
            plot.update(simulator)
            canvas.draw_idle()
        if future is None:
            action, pending_action = pending_action, None
            if action == "reset":
                simulator.reset()
                plot.update(simulator)
                canvas.draw_idle()
                status.set("Reset · same seeded replay starts again; slider unchanged")
            elif action == "export":
                destination = (Path(__file__).resolve().parents[1] / "runs/sim" /
                               datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                try:
                    simulator.export(destination)
                    plot.save(destination / "live_comparison.png")
                    status.set(f"Exported {destination}")
                except (OSError, ValueError) as error:
                    messagebox.showerror("Export failed", str(error))
            if running and time.perf_counter() >= due:
                future = executor.submit(simulator.step, battery_value.get())
                due = time.perf_counter() + 1 / rate.get()
        root.after(35, tick)

    def close():
        executor.shutdown(wait=False, cancel_futures=True)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.after(35, tick)
    root.mainloop()
