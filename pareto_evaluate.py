"""Select Pareto policies on validation, then evaluate the frozen family on test."""
import json
import os
import platform
import hashlib
from copy import copy
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mplconfig"))

import torch

from adaptive_evaluate import parse_grid
from src.evaluation import evaluation_parser, load_evaluation, write_csv
from src.metrics import collect_logits, policy_predictions, policy_metrics
from src.model import model_summary
from src.pareto import (METHOD, COMPUTE_NOTE, evaluate_policy, mark_frontier,
                        plot_frontier, select_policies, sha256, static_policies)
from src.utils import write_json

DEFAULT_GRID = "0,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1"


def code_identity():
    root = Path(__file__).resolve().parent
    # Normalize Git line endings so manifests also work after a fresh clone.
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            for name in ("pareto_evaluate.py", "src/pareto.py", "src/model.py", "src/metrics.py",
                         "src/data.py", "src/evaluation.py", "src/utils.py", "adaptive_evaluate.py")}


def validate_manifest(manifest, checkpoint_hash, code_hashes, baseline_hash):
    if manifest.get("schema_version") != 1 or manifest.get("selection_split") != "validation":
        raise ValueError("Expected a version-1 validation policy manifest")
    if manifest.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint changed: select policies again on validation")
    if manifest.get("baseline_checkpoint_sha256") != baseline_hash:
        raise ValueError("Static baseline changed: restore it or start a new validation comparison")
    if manifest.get("compute_method") != METHOD or manifest.get("code_sha256") != code_hashes:
        raise ValueError("Evaluation code/method changed: regenerate the validation manifest")
    policies = manifest.get("policies", [])
    if not policies or len({p["id"] for p in policies}) != len(policies):
        raise ValueError("Manifest needs nonempty, uniquely identified policies")
    if any(reference not in policies for reference in static_policies()):
        raise ValueError("Manifest must include all three fixed static references")
    for policy in policies:
        if policy["kind"] == "static":
            if policy not in static_policies():
                raise ValueError("Invalid static reference")
        elif policy["kind"] == "adaptive":
            if policy["mode"] not in ("normal", "low-power"):
                raise ValueError("Invalid policy mode")
            if not 0 <= policy["threshold_1"] <= 1:
                raise ValueError("Invalid first threshold")
            if policy["mode"] == "normal" and not 0 <= policy["threshold_2"] <= 1:
                raise ValueError("Invalid second threshold")
        else:
            raise ValueError("Unknown policy kind")


def validation_candidates(model, loader, device, temperatures, grid1, grid2):
    logits, labels = collect_logits(model, loader, device)
    costs = torch.tensor([r["macs_per_window"] for r in model_summary(model)["exits"]], dtype=torch.int64)
    rows = []
    for mode in ("normal", "low-power"):
        for t1 in grid1:
            for t2 in grid2 if mode == "normal" else [None]:
                predictions, exits = policy_predictions(logits, t1, t2 or 0., temperatures, mode)
                report = policy_metrics(labels.numpy(), predictions.numpy(), exits)
                average_macs = costs[exits - 1].sum().item() / len(labels)
                rows.append({"id": f"{mode}_{t1}_{t2 if t2 is not None else 'off'}",
                             "kind": "adaptive", "mode": mode, "threshold_1": t1,
                             "threshold_2": t2, "depth": None,
                             "accuracy": report["accuracy"], "macro_f1": report["macro_f1"],
                             "samples": len(labels), "average_stages": report["average_stages"],
                             "average_macs": average_macs, "average_flops": 2 * average_macs,
                             **{f"{key}_percent": value for key, value in report["exit_percentages"].items()}})
    return rows


def save_results(directory, rows, metadata, details=None):
    rows = mark_frontier(rows)
    write_csv(directory / "operating_points.csv", rows)
    write_json(directory / "results.json", {**metadata, "operating_points": rows,
                                            "classification_reports": details or {}})
    title = f"HAR accuracy vs. FLOPs — {metadata['split']}"
    title += " (synthetic smoke only)" if metadata["smoke_run"] else " (exploratory, single checkpoint)"
    plot_frontier(rows, directory / "pareto.png", title)
    baseline = next(r for r in rows if r["id"] == "static_exit_3")
    comparisons = [{"policy": row["id"],
                    "accuracy_change_percentage_points": 100 * (row["accuracy"] - baseline["accuracy"]),
                    "compute_reduction_percent": 100 * (1 - row["average_flops"] / baseline["average_flops"])}
                   for row in rows if row["kind"] == "adaptive"]
    write_json(directory / "baseline_comparison.json", {
        "reference": "Static full-depth path from the frozen baseline checkpoint; unused heads skipped",
        "comparisons": comparisons})
    lines = [f"# Pareto evaluation: {metadata['split']}", "", COMPUTE_NOTE, "",
             "Each dot is an evaluated configuration. The frontier is empirical over these points; "
             "lines do not imply measured intermediate policies or statistical significance.", "",
             "| Configuration | Accuracy (%) | Average FLOPs/window | Non-dominated |",
             "| --- | ---: | ---: | --- |"]
    lines += [f"| {r['id']} | {100*r['accuracy']:.3f} | {r['average_flops']:.1f} | {r['pareto']} |" for r in rows]
    lines += ["", "## Interpretation and limits", "",
              f"The evaluated family spans {min(r['average_flops'] for r in rows):,.0f} to "
              f"{max(r['average_flops'] for r in rows):,.0f} FLOPs/window. "
              f"{sum(not r['pareto'] for r in rows)} of {len(rows)} configurations are dominated "
              "in this sample (equal-coordinate ties are retained).", "",
              "See baseline_comparison.json for accuracy differences (percentage points) and compute reductions "
              "relative to the fixed full-depth path. No policy is chosen using these test differences.", "",
              "The three static references use the frozen baseline checkpoint and skip unused heads. They are "
              "fixed-depth ablations, not separately trained or tuned static CNNs. A stronger independently "
              "trained static comparison and a compression experiment remain project work.", "",
              "The included checkpoint is exploratory and single-seed; its documentation discloses prior "
              "inspection of test subjects. This workflow prevents new test-threshold tuning but cannot "
              "make those subjects a fresh holdout. Small differences need repeated runs/uncertainty analysis.", "",
              "Synthetic or smoke-run results are only software checks, not evidence of HAR performance. "
              "Changing training, compression, preprocessing or calibration requires a new validation manifest."]
    if comparisons:
        nearest = min(comparisons, key=lambda c: (abs(c["accuracy_change_percentage_points"]),
                                                  -c["compute_reduction_percent"]))
        lines += ["", "## Descriptive matched-accuracy comparison", "",
                  f"Among the evaluated adaptive configurations, `{nearest['policy']}` is closest in accuracy "
                  f"to the full-depth static reference: {nearest['accuracy_change_percentage_points']:+.3f} "
                  f"percentage points, with {nearest['compute_reduction_percent']:.2f}% fewer calculated FLOPs. "
                  "This is a description of the plotted results, not a test-selected deployment policy "
                  "or evidence of statistical equivalence. Negative compute reduction means more work."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = evaluation_parser(__doc__)
    parser.set_defaults(split="validation")
    parser.add_argument("--policies", type=Path, help="Frozen validation manifest, required for test")
    parser.add_argument("--baseline-checkpoint", default="models/adaptive_har.pt",
                        help="Frozen checkpoint for static depth references; keep fixed across experiments")
    parser.add_argument("--thresholds-1", help=f"Validation grid (default {DEFAULT_GRID})")
    parser.add_argument("--thresholds-2", help=f"Validation grid (default {DEFAULT_GRID})")
    args = parser.parse_args()
    if args.split == "test" and (not args.policies or args.thresholds_1 or args.thresholds_2):
        parser.error("Test requires --policies and disallows threshold grids; select on validation first")
    if args.split == "validation" and args.policies:
        parser.error("--policies is only for frozen test evaluation")
    directory = Path(args.output_dir or f"runs/adaptive_har/pareto_{args.split}")
    if directory.exists() and any(directory.iterdir()):
        parser.error("Output directory is not empty; use a new --output-dir to preserve previous results")
    checkpoint_hash, code_hashes = sha256(args.checkpoint), code_identity()
    baseline_hash = sha256(args.baseline_checkpoint)
    manifest = None
    if args.policies:
        manifest = json.loads(args.policies.read_text(encoding="utf-8"))
        try:
            validate_manifest(manifest, checkpoint_hash, code_hashes, baseline_hash)
        except ValueError as error:
            parser.error(str(error))
    model, loader, device, checkpoint = load_evaluation(args)
    if not checkpoint.get("calibrated", False):
        parser.error("Checkpoint requires validation calibration first")
    temperatures = checkpoint["temperatures"]
    baseline_model, baseline_loader = model, loader
    if baseline_hash != checkpoint_hash:
        baseline_args = copy(args)
        baseline_args.checkpoint = args.baseline_checkpoint
        baseline_model, baseline_loader, _, baseline_checkpoint = load_evaluation(baseline_args)
        for key in ("source", "channels", "class_names", "train_subjects", "validation_subjects", "test_subjects", "sizes"):
            if baseline_checkpoint["data_metadata"][key] != checkpoint["data_metadata"][key]:
                parser.error(f"Baseline and adaptive checkpoint must use identical dataset splits: {key}")
    else:
        baseline_checkpoint = checkpoint
    metadata = {"schema_version": 1, "split": args.split,
                "checkpoint_sha256": checkpoint_hash, "checkpoint": str(args.checkpoint),
                "baseline_checkpoint": str(args.baseline_checkpoint), "baseline_checkpoint_sha256": baseline_hash,
                "baseline_data_metadata": baseline_checkpoint["data_metadata"],
                "code_sha256": code_hashes, "compute_method": METHOD, "compute_note": COMPUTE_NOTE,
                "data_metadata": checkpoint["data_metadata"], "temperatures": temperatures,
                "samples": len(loader.dataset), "device": str(device), "batch_size": args.batch_size,
                "threads": args.threads, "torch_version": str(torch.__version__),
                "python_version": platform.python_version(),
                "package_versions": {name: version(name) for name in ("numpy", "scikit-learn", "matplotlib")},
                "smoke_run": checkpoint["data_metadata"]["source"] != "uci_har"
                or checkpoint["training_config"]["max_train_batches"] is not None
                or checkpoint["training_config"]["max_val_batches"] is not None
                or baseline_checkpoint["training_config"]["max_train_batches"] is not None
                or baseline_checkpoint["training_config"]["max_val_batches"] is not None}
    directory.mkdir(parents=True, exist_ok=True)
    details = {}
    if args.split == "validation":
        grid1 = sorted(set(parse_grid(args.thresholds_1 or DEFAULT_GRID)))
        grid2 = sorted(set(parse_grid(args.thresholds_2 or DEFAULT_GRID)))
        rows = validation_candidates(model, loader, device, temperatures, grid1, grid2)
        policies = select_policies(rows) + static_policies()
        # Check selected cached estimates against genuine conditional execution.
        for policy in policies:
            static = policy["kind"] == "static"
            actual, report = evaluate_policy(baseline_model if static else model,
                                            baseline_loader if static else loader, device, policy, temperatures)
            details[policy["id"]] = report
            if policy["kind"] == "static":
                rows.append(actual)
            else:
                cached = next(r for r in rows if r["id"] == policy["id"])
                for key in ("accuracy", "average_flops", "exit1_percent", "exit2_percent", "final_percent"):
                    if abs(actual[key] - cached[key]) > 1e-5:
                        raise RuntimeError(f"Cached/actual routing mismatch for {policy['id']}: {key}")
        manifest = {"schema_version": 1, "selection_split": "validation",
                    "checkpoint_sha256": checkpoint_hash, "code_sha256": code_hashes,
                    "baseline_checkpoint_sha256": baseline_hash,
                    "compute_method": METHOD, "thresholds_1": grid1, "thresholds_2": grid2,
                    "selection_rule": "All non-dominated validation policies within each mode; "
                    "equal accuracy/cost ties take first in sorted grid. Always include three static depths.",
                    "policies": policies}
        write_json(directory / "policies.json", manifest)
        metadata["execution"] = "Full validation grid from cached logits; selected policies checked with actual execution"
        metadata["selected_policy_ids"] = [p["id"] for p in policies]
    else:
        rows = []
        for policy in manifest["policies"]:
            static = policy["kind"] == "static"
            row, report = evaluate_policy(baseline_model if static else model,
                                         baseline_loader if static else loader, device, policy, temperatures)
            rows.append(row)
            details[policy["id"]] = report
        metadata["execution"] = "Actual execution of every frozen policy; no test-based policy selection"
        metadata["manifest_sha256"] = sha256(args.policies)
        write_json(directory / "policies.json", manifest)
    save_results(directory, rows, metadata, details)
    print(f"Saved {len(rows)} {args.split} operating points, frontier, comparisons and provenance to {directory}")


if __name__ == "__main__":
    main()
