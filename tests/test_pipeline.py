import tempfile
import unittest
import io
import zipfile
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import numpy as np
import torch

from src.data import _extract, channel_statistics, dataset_directory, load_split, make_loader, prepare_data, subject_masks
from src.metrics import classification_metrics, fit_temperatures, policy_predictions, threshold_sweep
from src.model import AdaptiveHAR, model_summary
from src.training import joint_loss, run_epoch
from tests.fixtures import write_uci_fixture


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(11)

    def test_shapes_parameters_and_summary(self):
        model = AdaptiveHAR().eval()
        with torch.no_grad():
            outputs = model(torch.randn(4, 9, 128))
        self.assertEqual([tuple(x.shape) for x in outputs], [(4, 6)] * 3)
        summary = model_summary(model)
        self.assertEqual(summary["total_parameters"], 2486)
        self.assertEqual(summary["shapes"]["stem"], [1, 4, 64])
        self.assertEqual(summary["shapes"]["stage1"], [1, 4, 64])
        self.assertEqual(summary["shapes"]["stage2"], [1, 7, 32])
        self.assertEqual(summary["shapes"]["stage3"], [1, 7, 16])
        self.assertEqual(summary["exits"][-1]["parameters_up_to_exit"], summary["total_parameters"])
        self.assertLess(summary["exits"][0]["macs_per_window"], summary["exits"][1]["macs_per_window"])
        self.assertFalse(model.training)

    def test_load_batch_forward_backward_and_no_normalization_leakage(self):
        with tempfile.TemporaryDirectory() as root:
            directory = write_uci_fixture(root)
            data = prepare_data(root, val_subjects=[5])
            raw, labels, subjects = load_split(directory, "train")
            mean, std = channel_statistics(raw[subjects != 5])
            np.testing.assert_allclose(data.metadata["normalization"]["mean"], mean)
            np.testing.assert_allclose(data.metadata["normalization"]["std"], std)
            np.testing.assert_allclose(data.train.tensors[0].mean(dim=(0, 2)).numpy(), 0, atol=2e-6)
            np.testing.assert_allclose(data.train.tensors[0].std(dim=(0, 2), unbiased=False).numpy(), 1, atol=2e-6)
            self.assertGreater(float(data.validation.tensors[0].mean()), .5)
            self.assertGreater(float(data.test.tensors[0].mean()), 1.)
            self.assertEqual(data.metadata["validation_subjects"], [5])
            self.assertFalse(set(data.metadata["train_subjects"]) & set(data.metadata["validation_subjects"]))
            self.assertFalse(set(data.metadata["train_subjects"]) & set(data.metadata["test_subjects"]))
            self.assertEqual(int(labels.min()), 0)
            self.assertEqual(int(labels.max()), 5)
            x, y = next(iter(make_loader(data.train, 8)))
            model = AdaptiveHAR()
            before = model.stem[0].weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss, _ = joint_loss(model(x), y)
            loss.backward()
            for head in [model.exit1, model.exit2, model.final]:
                self.assertIsNotNone(head[-1].weight.grad)
                self.assertTrue(torch.isfinite(head[-1].weight.grad).all())
            optimizer.step()
            self.assertFalse(torch.equal(before, model.stem[0].weight))
            # The checkpoint's statistics must override any refit during evaluation.
            restored = prepare_data(root, val_subjects=[5], normalization=data.metadata["normalization"])
            torch.testing.assert_close(restored.test.tensors[0], data.test.tensors[0])

    def test_selected_architecture_restores_without_architecture_switching(self):
        model = AdaptiveHAR().eval()
        restored = AdaptiveHAR(**model.config).eval()
        restored.load_state_dict(model.state_dict())
        x = torch.randn(4, 9, 128)
        with torch.no_grad():
            for expected, actual in zip(model(x), restored(x)):
                torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            AdaptiveHAR(model_size="unsupported")
        with self.assertRaises(ValueError):
            AdaptiveHAR(final_mode="unsupported")

    def test_exit_parameter_allocation(self):
        summary = model_summary(AdaptiveHAR())
        self.assertEqual(summary["total_parameters"], 2486)
        for exit_report, target in zip(summary["exits"][:2], [.30, .70]):
            self.assertLess(abs(exit_report["parameters_up_to_exit"] / 2486 - target), .005)

    def test_subject_split_determinism_and_invalid_splits(self):
        subjects = np.repeat(np.arange(1, 8), 10)
        first = subject_masks(subjects, seed=42)
        second = subject_masks(subjects, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(set(subjects[first[0]]) & set(subjects[first[1]]))
        for ids in [[], [99], list(range(1, 8))]:
            with self.assertRaises(ValueError):
                subject_masks(subjects, val_subjects=ids)

    def test_adaptive_skips_stages_and_matches_full_logits(self):
        model = AdaptiveHAR().eval()
        x = torch.randn(5, 9, 128)
        with torch.no_grad():
            logits = model(x)
        with patch.object(model.stage2, "forward", side_effect=AssertionError("stage2 ran")), \
             patch.object(model.stage3, "forward", side_effect=AssertionError("stage3 ran")):
            result = model.adaptive_forward(x, 0., 0.)
            torch.testing.assert_close(result.logits, logits[0])
            self.assertTrue((result.exit_indices == 1).all())
        with patch.object(model.stage3, "forward", side_effect=AssertionError("stage3 ran")):
            result = model.adaptive_forward(x, 1., 1., mode="low-power")
            torch.testing.assert_close(result.logits, logits[1])
            self.assertTrue((result.exit_indices == 2).all())
            result = model.adaptive_forward(x, 1., 0.)
            self.assertTrue((result.exit_indices == 2).all())
        result = model.adaptive_forward(x, 1., 1.)
        torch.testing.assert_close(result.logits, logits[2])
        self.assertTrue((result.exit_indices == 3).all())
        model.train()
        with self.assertRaises(RuntimeError):
            model.adaptive_forward(x, .5, .5)

    def test_mixed_batch_routes_compactly_in_original_order(self):
        model = AdaptiveHAR().eval()
        x = torch.randn(8, 9, 128)
        temperatures = [1.2, .8, 1.1]
        with torch.no_grad():
            logits = model(x)
        confidences = (logits[0] / temperatures[0]).softmax(1).amax(1)
        threshold_1 = float(confidences.median())
        threshold_2 = float((logits[1] / temperatures[1]).softmax(1).amax(1).median())
        batches = []
        handle = model.stage2.register_forward_pre_hook(lambda module, inputs: batches.append(len(inputs[0])))
        try:
            actual = model.adaptive_forward(x, threshold_1, threshold_2, temperatures=temperatures)
        finally:
            handle.remove()
        expected, exits = policy_predictions(logits, threshold_1, threshold_2, temperatures)
        torch.testing.assert_close(actual.predictions, expected)
        torch.testing.assert_close(actual.exit_indices, exits)
        self.assertTrue(0 < batches[0] < len(x))
        for i in range(len(x)):
            exit_index = int(exits[i]) - 1
            torch.testing.assert_close(actual.logits[i], logits[exit_index][i] / temperatures[exit_index], atol=1e-6, rtol=1e-4)

    def test_calibration_and_policy_sweep(self):
        # These have the same inference-tensor provenance as collect_logits.
        with torch.inference_mode():
            labels = torch.arange(60) % 6
            logits = tuple(torch.randn(60, 6) * 5 for _ in range(3))
        temperatures, diagnostics = fit_temperatures(logits, labels, max_steps=30)
        self.assertTrue(all(t > 0 for t in temperatures))
        self.assertTrue(all(d["validation_nll_after"] <= d["validation_nll_before"] + 1e-6 for d in diagnostics))
        rows = threshold_sweep(logits, labels, [0., 1.], [0., 1.], temperatures)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["average_stages"], 1.)
        self.assertEqual(rows[-1]["average_stages"], 3.)
        for row in rows:
            self.assertAlmostEqual(row["exit1_percent"] + row["exit2_percent"] + row["final_percent"], 100., places=4)
        low_power = threshold_sweep(logits, labels, [0., 1.], [0., .5, 1.], temperatures, "low-power")
        self.assertEqual(len(low_power), 2)
        self.assertTrue(all(row["final_percent"] == 0 for row in low_power))

    def test_nested_archive_extraction_and_path_safety(self):
        with tempfile.TemporaryDirectory() as staging, tempfile.TemporaryDirectory() as root:
            directory = write_uci_fixture(staging, windows_per_subject=6)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as inner:
                for path in directory.rglob("*"):
                    if path.is_file():
                        inner.write(path, path.relative_to(staging))
            with zipfile.ZipFile(Path(root) / "uci_har_complete.zip", "w") as outer:
                outer.writestr("UCI HAR Dataset.zip", buffer.getvalue())
                outer.writestr("UCI HAR Dataset.names", "Full archive metadata retained")
            extracted = dataset_directory(root, download=True)
            self.assertTrue((extracted / "test/Inertial Signals/body_acc_x_test.txt").exists())
            self.assertTrue((Path(root) / "UCI HAR Dataset.names").exists())
            self.assertTrue((Path(root) / "download.json").exists())
            bad_archive = Path(root) / "unsafe.zip"
            with zipfile.ZipFile(bad_archive, "w") as archive:
                archive.writestr("../outside.txt", "unsafe")
            with self.assertRaises(ValueError):
                _extract(bad_archive, Path(root))

    def test_class_metrics_include_missing_classes_and_confusion_axes(self):
        result = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertEqual(result["accuracy"], .75)
        self.assertEqual(result["confusion_matrix"][0][:2], [1, 1])
        self.assertEqual(result["confusion_matrix"][1][:2], [0, 2])
        self.assertEqual(result["per_class"]["WALKING"]["recall"], .5)
        self.assertEqual(result["per_class"]["LAYING"]["support"], 0)

    def test_real_dataset_batch_if_available(self):
        root = Path(__file__).resolve().parents[1] / "data"
        if not (root / "UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt").exists():
            self.skipTest("Official UCI archive has not been downloaded; synthetic file-based smoke test still runs")
        bundle = prepare_data(root)
        model = AdaptiveHAR()
        loader = make_loader(bundle.train, batch_size=8)
        result = run_epoch(model, loader, torch.device("cpu"), optimizer=torch.optim.AdamW(model.parameters()), max_batches=1)
        self.assertEqual(result["samples"], 8)


if __name__ == "__main__":
    unittest.main()
