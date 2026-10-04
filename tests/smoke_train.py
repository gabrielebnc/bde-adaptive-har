"""Exercise all CLI paths offline; artifacts are explicitly synthetic smoke data."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from tests.fixtures import write_uci_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="runs/synthetic_smoke")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.output_dir).resolve()
    data_dir, run_dir = root / "data", root / "model"
    if (run_dir / "best.pt").exists():
        parser.error("Smoke checkpoint already exists; choose a new --output-dir")
    write_uci_fixture(data_dir, windows_per_subject=24)
    def run(script, arguments):
        subprocess.run([sys.executable, str(project / script), *map(str, arguments)], cwd=project, check=True)
    run("train.py", ["--data-dir", data_dir, "--output-dir", run_dir, "--epochs", 2,
                     "--batch-size", 8, "--max-train-batches", 2, "--max-val-batches", 1,
                     "--calibration-steps", 20])
    common = ["--checkpoint", run_dir / "best.pt", "--data-dir", data_dir, "--batch-size", 8]
    run("evaluate.py", common + ["--split", "test"])
    for mode in ["normal", "low-power"]:
        run("adaptive_evaluate.py", common + ["--sweep", "--mode", mode,
                                              "--thresholds-1", "0,0.5,1", "--thresholds-2", "0,0.5,1"])
        # Boundary thresholds are routing checks, not recommended policies.
        run("adaptive_evaluate.py", common + ["--split", "test", "--mode", mode,
                                              "--threshold-1", 1, "--threshold-2", 0])
    checkpoint_reports = json.loads((run_dir / "evaluation_test/metrics.json").read_text())
    assert checkpoint_reports["data_source"] == "synthetic_test_fixture"
    assert checkpoint_reports["smoke_run"]
    assert (run_dir / "history.png").exists()
    for mode in ["normal", "low-power"]:
        report = json.loads((run_dir / f"adaptive_test_{mode}/policy_metrics.json").read_text())
        assert report["exit_percentages"]["final"] == 0
        assert report["average_stages"] == 2
        assert (run_dir / f"adaptive_validation_{mode}/threshold_sweep.csv").exists()
    print(f"All smoke CLI checks passed. Synthetic artifacts: {root}")


if __name__ == "__main__":
    main()
