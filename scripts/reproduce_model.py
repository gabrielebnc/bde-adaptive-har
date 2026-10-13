"""Download UCI HAR, rebuild the selected model, evaluate, and check its weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]


def model_state_digest(state):
    """Hash named tensor values independently of checkpoint paths/serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                            separators=(",", ":")).encode()
        digest.update(header + b"\0")
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(values.astype(values.dtype.newbyteorder("<"), copy=False).tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data",
                        help="Dataset parent directory; downloads here if missing (default: data)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/adaptive_har_retrain",
                        help="Fresh directory for the rebuilt checkpoint and reports (default: runs/adaptive_har_retrain)")
    parser.add_argument("--require-exact", action="store_true",
                        help="Fail if rebuilt tensors or calibration differ from the recorded reference")
    args = parser.parse_args()
    data, run = args.data_dir.resolve(), args.output_dir.resolve()
    if (run / "best.pt").exists():
        parser.error("Output checkpoint already exists; choose a new --output-dir")
    recipe = json.loads((PROJECT / "models/adaptive_har.recipe.json").read_text())
    training_args = []
    for key, value in recipe["training"].items():
        training_args += ["--" + key.replace("_", "-"),
                          *map(str, value if isinstance(value, list) else [value])]

    def invoke(script, options):
        subprocess.run([sys.executable, str(PROJECT / script), *options], cwd=PROJECT, check=True)

    print("Training the recorded CPU recipe from random initialization.", flush=True)
    invoke("train.py", ["--download", "--data-dir", str(data), "--output-dir", str(run), *training_args])
    invoke("scripts/evaluate_run.py", ["--run-dir", str(run), "--data-dir", str(data),
                                      "--device", "cpu", "--threads", str(recipe["training"]["threads"])])

    import torch
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    summary = json.loads((run / "training_summary.json").read_text())
    actual = {"model_state_sha256": model_state_digest(checkpoint["model_state"]),
              "temperatures": checkpoint["temperatures"], "epoch": checkpoint["epoch"],
              "epochs_completed": summary["epochs_completed"]}
    matches = actual == recipe["reference"]
    report = {"matches_reference": matches, "actual": actual, "reference": recipe["reference"]}
    (run / "reproduction_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Rebuilt checkpoint: {run / 'best.pt'}")
    print(f"Exact reference match: {'yes' if matches else 'no'}")
    print(f"Verification report: {run / 'reproduction_check.json'}")
    if not matches:
        print("Training completed; different platforms or package versions can change the result. "
              "See requirements-reproduce.txt and docs/model.md.")
        if args.require_exact:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
