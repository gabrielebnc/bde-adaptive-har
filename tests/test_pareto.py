import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from pareto_evaluate import validate_manifest
from src.data import prepare_data
from src.model import AdaptiveHAR, model_summary
from src.pareto import (METHOD, count_macs, evaluate_policy, frontier_indices,
                        select_policies, static_forward, static_policies)
from tests.fixtures import write_uci_fixture


class ParetoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dominance_ties_and_invalid_metrics(self):
        rows = [{"average_flops": cost, "accuracy": acc} for cost, acc in
                [(10, .6), (20, .6), (30, .8), (20, .7), (20, .7), (40, .75)]]
        self.assertEqual(frontier_indices(rows), [0, 2, 3, 4])
        self.assertEqual(frontier_indices([]), [])
        for cost, accuracy in [(float("nan"), .5), (0, .5), (10, float("inf")), (10, 90)]:
            with self.assertRaises(ValueError):
                frontier_indices([{"average_flops": cost, "accuracy": accuracy}])

    def test_selection_preserves_modes_and_deduplicates_equal_points(self):
        rows = []
        for mode in ("normal", "low-power"):
            for i, (cost, acc) in enumerate([(10, .7), (10, .7), (20, .6), (30, .8)]):
                rows.append({"id": f"{mode}_{i}", "kind": "adaptive", "mode": mode,
                             "threshold_1": .5, "threshold_2": .8, "depth": None,
                             "average_flops": cost, "accuracy": acc})
        self.assertEqual([r["id"] for r in select_policies(rows)],
                         ["normal_0", "normal_3", "low-power_0", "low-power_3"])

    def test_static_execution_skips_heads_and_matches_predictions(self):
        model = AdaptiveHAR().eval()
        x = torch.randn(4, 9, 128)
        with torch.inference_mode():
            expected = model(x)
            with patch.object(model.exit1, "forward", side_effect=AssertionError("unused head")), \
                 patch.object(model.exit2, "forward", side_effect=AssertionError("unused head")):
                with count_macs(model) as counts:
                    actual = static_forward(model, x, 3)
            torch.testing.assert_close(actual, expected[2])
        full_cost = model_summary(model)["exits"][2]["macs_per_window"]
        # Earlier heads cost channels * six classes: 4*6 + 7*6.
        self.assertEqual(counts["macs"], len(x) * (full_cost - 66))
        with patch.object(model.stage2, "forward", side_effect=AssertionError("unused stage")):
            with torch.inference_mode():
                torch.testing.assert_close(static_forward(model, x, 1), expected[0])

    def test_mixed_routing_cost_uses_actual_sample_counts(self):
        torch.manual_seed(8)
        model = AdaptiveHAR().eval()
        x, y = torch.randn(9, 9, 128), torch.arange(9) % 6
        with torch.inference_mode():
            logits = model(x)
            threshold = float(logits[0].softmax(1).amax(1).median())
            outputs = model.adaptive_forward(x, threshold, 0.)
        loader = DataLoader(TensorDataset(x, y), batch_size=9)
        policy = {"id": "mixed", "kind": "adaptive", "mode": "normal",
                  "threshold_1": threshold, "threshold_2": 0., "depth": None}
        row, _ = evaluate_policy(model, loader, torch.device("cpu"), policy, [1., 1., 1.])
        costs = torch.tensor([r["macs_per_window"] for r in model_summary(model)["exits"]])
        expected = costs[outputs.exit_indices - 1].sum().item() / len(x)
        self.assertEqual(row["average_macs"], expected)
        self.assertTrue(0 < row["exit1_percent"] < 100)
        self.assertEqual(row["final_percent"], 0)

    def test_manifest_rejects_checkpoint_or_method_changes(self):
        manifest = {"schema_version": 1, "selection_split": "validation",
                    "checkpoint_sha256": "abc", "code_sha256": {"code": "123"},
                    "baseline_checkpoint_sha256": "base",
                    "compute_method": METHOD, "policies": static_policies()}
        validate_manifest(manifest, "abc", {"code": "123"}, "base")
        for key, value in [("selection_split", "test"), ("checkpoint_sha256", "other"),
                           ("compute_method", "other"), ("code_sha256", {}),
                           ("baseline_checkpoint_sha256", "other")]:
            with self.assertRaises(ValueError):
                validate_manifest({**manifest, key: value}, "abc", {"code": "123"}, "base")

    def test_cli_validation_then_frozen_test(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_uci_fixture(tmp / "data", windows_per_subject=6)
            bundle = prepare_data(tmp / "data", val_subjects=[5])
            model = AdaptiveHAR()
            checkpoint = tmp / "fixture.pt"
            torch.save({"model_config": model.config, "model_state": model.state_dict(),
                        "data_metadata": bundle.metadata, "calibrated": True,
                        "temperatures": [1., 1., 1.], "epoch": 0,
                        "training_config": {"max_train_batches": 1, "max_val_batches": 1}}, checkpoint)
            baseline = tmp / "baseline.pt"
            payload = torch.load(checkpoint, weights_only=True)
            payload["model_state"]["final.3.weight"].zero_()
            payload["model_state"]["final.3.bias"].zero_()
            payload["model_state"]["final.3.bias"][0] = 100.
            torch.save(payload, baseline)
            common = [sys.executable, str(root / "pareto_evaluate.py"), "--checkpoint", str(checkpoint),
                      "--baseline-checkpoint", str(baseline),
                      "--data-dir", str(tmp / "data"), "--device", "cpu"]

            def run(*args):
                return subprocess.run(common + list(args), cwd=root, text=True, capture_output=True)

            validation = run("--output-dir", str(tmp / "val"), "--thresholds-1", "0,1", "--thresholds-2", "0,1")
            self.assertEqual(validation.returncode, 0, validation.stderr)
            manifest_path = tmp / "val/policies.json"
            result = run("--split", "test", "--policies", str(manifest_path), "--output-dir", str(tmp / "test"))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((tmp / "test/results.json").read_text())
            manifest = json.loads(manifest_path.read_text())
            self.assertTrue(report["smoke_run"])
            self.assertNotEqual(report["checkpoint_sha256"], report["baseline_checkpoint_sha256"])
            confusion = report["classification_reports"]["static_exit_3"]["confusion_matrix"]
            self.assertTrue(all(row[0] == sum(row) for row in confusion))
            self.assertEqual([r["id"] for r in report["operating_points"]],
                             [r["id"] for r in manifest["policies"]])
            self.assertTrue((tmp / "test/pareto.svg").exists())
            self.assertTrue((tmp / "test/baseline_comparison.json").exists())
            self.assertNotEqual(run("--split", "test").returncode, 0)
            self.assertNotEqual(run("--split", "test", "--policies", str(manifest_path),
                                    "--thresholds-1", "0,1").returncode, 0)
            self.assertNotEqual(run("--output-dir", str(tmp / "val")).returncode, 0)


if __name__ == "__main__":
    unittest.main()
