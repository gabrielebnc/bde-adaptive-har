import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluation import artifact_directory, load_evaluation
from src.metrics import EXIT_NAMES, classification_metrics, collect_logits
from src.model import AdaptiveHAR, model_summary
from src.training import DEFAULT_LOSS_WEIGHTS


PROJECT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT / "models/adaptive_har.pt"
CHECKPOINT_SHA256 = "3c272d8a9856e8ec1c4d73079e350e60b8c3fde36d0a2fe870be8b3bbada2896"


class SelectedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_packaged_checkpoint_is_the_selected_model_and_matches_defaults(self):
        self.assertEqual(hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(), CHECKPOINT_SHA256)
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model = AdaptiveHAR().eval()
        self.assertEqual(model.config, checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"])
        self.assertEqual(model_summary(model)["total_parameters"], 2486)
        self.assertEqual(checkpoint["epoch"], 41)
        self.assertTrue(checkpoint["calibrated"])
        self.assertEqual(checkpoint["training_config"]["lr"], .001)
        self.assertEqual(checkpoint["training_config"]["loss_weights"], list(DEFAULT_LOSS_WEIGHTS))
        self.assertEqual(len(checkpoint["temperatures"]), 3)
        self.assertTrue(all(t > 0 for t in checkpoint["temperatures"]))
        with torch.inference_mode():
            self.assertEqual([tuple(v.shape) for v in model(torch.randn(4, 9, 128))], [(4, 6)] * 3)

    def test_packaged_model_reports_do_not_pollute_the_models_directory(self):
        args = SimpleNamespace(checkpoint=str(CHECKPOINT), output_dir=None)
        self.assertEqual(artifact_directory(args, "evaluation_test"), PROJECT / "runs/adaptive_har/evaluation_test")
        args.output_dir = "custom_reports"
        self.assertEqual(artifact_directory(args, "evaluation_test"), Path("custom_reports"))
        args = SimpleNamespace(checkpoint="runs/retrained/best.pt", output_dir=None)
        self.assertEqual(artifact_directory(args, "evaluation_test"), Path("runs/retrained/evaluation_test"))

    def test_packaged_checkpoint_matches_saved_real_data_metrics_if_available(self):
        data = PROJECT / "data"
        if not (data / "UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt").exists():
            self.skipTest("UCI HAR is not downloaded; checkpoint loading is tested separately")
        for split in ("validation", "test"):
            with self.subTest(split=split):
                args = SimpleNamespace(checkpoint=str(CHECKPOINT), data_dir=str(data), split=split,
                                       device="cpu", threads=2, batch_size=64, num_workers=0)
                model, loader, device, _ = load_evaluation(args)
                logits, labels = collect_logits(model, loader, device)
                saved = json.loads((PROJECT / f"docs/results/{split}.json").read_text())
                self.assertEqual(len(labels), saved["samples"])
                for name, values in zip(EXIT_NAMES, logits):
                    actual = classification_metrics(labels.numpy(), values.argmax(1).numpy())
                    self.assertEqual(actual["accuracy"], saved["exits"][name]["accuracy"])
                    self.assertEqual(actual["macro_f1"], saved["exits"][name]["macro_f1"])


if __name__ == "__main__":
    unittest.main()
