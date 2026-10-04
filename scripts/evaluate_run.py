"""Evaluate a completed run, select adaptive policies on validation, then test."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    common = ["--checkpoint", str(run / "best.pt"), "--data-dir", str(Path(args.data_dir).resolve()),
              "--device", args.device, "--threads", str(args.threads)]

    def invoke(script, extra):
        subprocess.run([sys.executable, str(project / script), *common, *extra], cwd=project, check=True)

    for split in ("validation", "test"):
        invoke("evaluate.py", ["--split", split])
    policies = []
    for mode in ("normal", "low-power"):
        invoke("adaptive_evaluate.py", ["--split", "validation", "--mode", mode, "--sweep"])
        sweep = json.loads((run / f"adaptive_validation_{mode}/threshold_sweep.json").read_text())
        selected = min(sweep["policies"], key=lambda p: (-p["accuracy"], p["average_macs"], -p["macro_f1"]))
        extra = ["--split", "test", "--mode", mode, "--threshold-1", str(selected["threshold_1"])]
        if mode == "normal":
            extra += ["--threshold-2", str(selected["threshold_2"])]
        invoke("adaptive_evaluate.py", extra)
        policies.append(selected)
    (run / "selected_policies.json").write_text(json.dumps({
        "selection": "Maximum validation accuracy; ties by minimum MACs, then maximum macro F1",
        "policies": policies,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
