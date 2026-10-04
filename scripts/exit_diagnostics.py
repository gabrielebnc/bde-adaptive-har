"""Count errors corrected and introduced between heads of one saved checkpoint."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import artifact_directory, load_evaluation
from src.metrics import collect_logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/adaptive_har.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    args.num_workers = 0
    reports = {}
    for split in ("validation", "test"):
        args.split = split
        model, loader, device, checkpoint = load_evaluation(args)
        logits, labels = collect_logits(model, loader, device)
        correct = [values.argmax(1) == labels for values in logits]
        reports[split] = {}
        for earlier, later in ((0, 1), (1, 2)):
            a, b = correct[earlier], correct[later]
            fixed = int((~a & b).sum())
            introduced = int((a & ~b).sum())
            reports[split][f"{earlier + 1}_to_{later + 1}"] = {
                "samples": len(labels), "errors_corrected": fixed,
                "new_errors": introduced, "net_corrected": fixed - introduced,
                "accuracy_change_percentage_points": 100 * (fixed - introduced) / len(labels),
            }
    report = {"checkpoint": args.checkpoint, "checkpoint_epoch": checkpoint["epoch"],
              "model_config": checkpoint["model_config"], "splits": reports,
              "purpose": "Descriptive evaluation; no checkpoint or policy selection"}
    output = artifact_directory(args, "exit_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    (output / "corrections.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
