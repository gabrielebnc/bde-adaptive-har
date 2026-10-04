import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.model import AdaptiveHAR, model_summary
from src.metrics import policy_predictions
from src.refinement import refinement_candidate, refinement_loss, train_refinement
from src.curriculum import fit_early_heads
from src.training import train_model


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_zero_correction_matches_exit2_without_extra_parameters(self):
        model = AdaptiveHAR(final_mode="residual").eval()
        x = torch.randn(8, 9, 128)
        with torch.no_grad():
            logits = model(x)
        self.assertTrue(torch.equal(logits[1], logits[2]))
        self.assertEqual(model_summary(model)["total_parameters"], 2486)
        restored = AdaptiveHAR(**model.config)
        restored.load_state_dict(model.state_dict())
        restored.eval()
        torch.testing.assert_close(restored(x)[2], logits[2])

    def test_correct_decision_damage_is_penalized_and_teacher_detached(self):
        base = torch.tensor([[3., 0., 0.], [0., 3., 0.]], requires_grad=True)
        final = torch.tensor([[0., 3., 0.], [0., 3., 0.]], requires_grad=True)
        labels = torch.tensor([0, 0])  # Only the first earlier decision is correct.
        objective, ce, damage = refinement_loss(base, final, labels, 1.)
        self.assertAlmostEqual(damage.item(), 3.)
        self.assertGreater(objective.item(), ce.item())
        objective.backward()
        self.assertIsNone(base.grad)
        self.assertLess(final.grad[0, 0].item(), 0.)
        self.assertGreater(final.grad[0, 1].item(), 0.)

    def test_better_cross_entropy_cannot_select_lower_accuracy(self):
        baseline = {"exit1_accuracy": .75, "exit2_accuracy": .80}
        best = (.80, -.2)
        self.assertFalse(refinement_candidate({"final_accuracy": .79, "loss": .01}, baseline, best))
        self.assertTrue(refinement_candidate({"final_accuracy": .81, "loss": .3}, baseline, best))
        self.assertTrue(refinement_candidate({"final_accuracy": .80, "loss": .1}, baseline, best))
        self.assertFalse(refinement_candidate({"final_accuracy": .80, "loss": .4}, baseline, best))

    def test_residual_adaptive_compaction_matches_cached_policy(self):
        model = AdaptiveHAR(final_mode="residual").eval()
        with torch.no_grad():
            model.final[-1].weight.normal_(0., .1)
        x, temps = torch.randn(12, 9, 128), [1.2, .8, 1.1]
        with torch.no_grad():
            logits = model(x)
        t1 = float((logits[0] / temps[0]).softmax(1).amax(1).median())
        t2 = float((logits[1] / temps[1]).softmax(1).amax(1).median())
        predictions, exits = policy_predictions(logits, t1, t2, temps)
        actual = model.adaptive_forward(x, t1, t2, temperatures=temps)
        self.assertTrue(torch.equal(actual.exit_indices, exits))
        self.assertTrue(torch.equal(actual.predictions, predictions))
        for i in range(3):
            indices = exits == i + 1
            torch.testing.assert_close(actual.logits[indices], logits[i][indices] / temps[i])

    def test_training_keeps_prefix_and_respects_validation_accuracy_floor(self):
        model = AdaptiveHAR(final_mode="residual").eval()
        with torch.no_grad():
            for head in (model.exit1[-1], model.exit2[-1]):
                head.weight.zero_()
                head.bias.zero_()
                head.bias[0] = 2.
        frozen = {n: v.clone() for n, v in model.state_dict().items()
                  if n.split(".")[0] not in {"stage3", "final"}}
        source = {"model_config": model.config, "epoch": 68, "temperatures": [.5, .7, .8],
                  "calibration": [{"temperature": t} for t in [.5, .7, .8]],
                  "data_metadata": {"source": "synthetic_test_fixture"}}
        config = {"source_checkpoint": "fixture.pt", "lr": .001, "weight_decay": .0001,
                  "epochs": 2, "patience": 2, "loss_weights": [.2, .3, .5],
                  "protection_weight": 1., "calibration_steps": 10}
        batches = [(torch.randn(8, 9, 128), torch.zeros(8, dtype=torch.long))]
        with tempfile.TemporaryDirectory() as root:
            result = train_refinement(model, batches, batches, torch.device("cpu"), root, source, config)
            checkpoint = torch.load(Path(root) / "best.pt", weights_only=True)
            self.assertGreaterEqual(result["best_validation_accuracy"], result["baseline_validation"]["exit2_accuracy"])
            self.assertEqual(checkpoint["temperatures"][:2], [.5, .7])
            self.assertTrue(result["frozen_state_verified"])
            for name, value in frozen.items():
                self.assertTrue(torch.equal(checkpoint["model_state"][name], value), name)

    def test_curriculum_preserves_full_model_and_its_calibration(self):
        model = AdaptiveHAR().eval()
        x, y = torch.randn(8, 9, 128), torch.arange(8) % 6
        frozen = {n: v.clone() for n, v in model.state_dict().items()
                  if n.split(".")[0] not in {"exit1", "exit2"}}
        final_before = model(x)[2].detach().clone()
        source = {"model_config": model.config, "model_state": model.state_dict(), "epoch": 23,
                  "temperatures": [.5, .7, .8], "calibrated": True,
                  "calibration": [{"temperature": t} for t in [.5, .7, .8]],
                  "data_metadata": {"source": "synthetic_test_fixture"}}
        config = {"source_checkpoint": "fixture.pt", "lr": .01, "weight_decay": .0001,
                  "epochs": 2, "patience": 2, "calibration_steps": 10}
        with tempfile.TemporaryDirectory() as root:
            result = fit_early_heads(model, [(x, y)], [(x, y)], torch.device("cpu"), root, source, config)
            ckpt = torch.load(Path(root) / "best.pt", weights_only=True)
            self.assertTrue(result["frozen_state_verified"])
            self.assertEqual(ckpt["temperatures"][2], .8)
            for n, value in frozen.items():
                self.assertTrue(torch.equal(ckpt["model_state"][n], value), n)
            model.eval()
            self.assertTrue(torch.equal(model(x)[2], final_before))

    def test_accuracy_checkpoint_selection_rejects_loss_only_improvement(self):
        def metrics(loss, accuracy):
            return {"loss": loss, "samples": 8,
                    **{f"{name}_loss": loss for name in ("exit1", "exit2", "final")},
                    **{f"{name}_accuracy": accuracy for name in ("exit1", "exit2", "final")}}
        calls = [metrics(.3, .8), metrics(.1, .85), metrics(.3, .8), metrics(.09, .84),
                 metrics(.3, .8), metrics(.11, .86)]
        config = {"lr": .001, "weight_decay": .0001, "epochs": 3, "patience": 3,
                  "min_delta": 0., "loss_weights": [.2, .3, .5], "max_train_batches": None,
                  "max_val_batches": None, "calibration_steps": 5, "checkpoint_metric": "final-accuracy"}
        model = AdaptiveHAR()
        logits, labels = (torch.zeros(8, 6),) * 3, torch.arange(8) % 6
        with tempfile.TemporaryDirectory() as root, \
             patch("src.training.run_epoch", side_effect=calls), \
             patch("src.training.collect_logits", return_value=(logits, labels)):
            result = train_model(model, [], [], torch.device("cpu"), root,
                                 {"source": "synthetic_test_fixture"}, config)
            self.assertEqual(result["best_epoch"], 3)
            self.assertAlmostEqual(result["best_validation_loss"], .11)


if __name__ == "__main__":
    unittest.main()
