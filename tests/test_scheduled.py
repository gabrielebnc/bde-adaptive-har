import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.metrics import EXIT_NAMES
from src.model import AdaptiveHAR
from src.scheduled_training import candidate_score, scheduled_weights, train_scheduled


class ScheduledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_schedule_starts_final_only_and_reaches_target(self):
        target = [.2, .3, .5]
        self.assertEqual(scheduled_weights(1, 30, target), (0., 0., 1.))
        for epoch in (2, 15, 30, 60):
            weights = scheduled_weights(epoch, 30, target)
            self.assertAlmostEqual(sum(weights), 1.)
            self.assertGreaterEqual(weights[2], .5)
        for actual, wanted in zip(scheduled_weights(30, 30, target), target):
            self.assertAlmostEqual(actual, wanted)
        self.assertEqual(scheduled_weights(30, 30, target), scheduled_weights(60, 30, target))
        for epoch, ramp, weights in [(0, 30, target), (1, 1, target), (1, 30, [.1, .1, .1]),
                                      (1, 30, [float('nan'), .3, .7])]:
            with self.assertRaises(ValueError):
                scheduled_weights(epoch, ramp, weights)

    def test_selection_rejects_regression_and_nonmonotonic_exits(self):
        def validation(a, b, c, loss=.2):
            return {"exit1_accuracy": a, "exit2_accuracy": b, "final_accuracy": c, "loss": loss}
        weights = [.2, .3, .5]
        self.assertIsNone(candidate_score(validation(.8, .9, .89, .01), .88, weights))
        self.assertIsNone(candidate_score(validation(.85, .8, .92), .9, weights))
        self.assertIsNone(candidate_score(validation(.8, .85, .89), .9, weights))
        lower_early = candidate_score(validation(.7, .8, .92), .9, weights)
        higher_early = candidate_score(validation(.8, .9, .92), .9, weights)
        self.assertGreater(higher_early, lower_early)

    def fixture(self):
        model = AdaptiveHAR().eval()
        with torch.no_grad():
            for head in (model.exit1[-1], model.exit2[-1], model.final[-1]):
                head.weight.zero_()
                head.bias.zero_()
                head.bias[0] = 3.
        source = {"epoch": 47, "model_config": model.config,
                  "model_state": {n: v.clone() for n, v in model.state_dict().items()},
                  "data_metadata": {"source": "synthetic_test_fixture"},
                  "temperatures": [.5, .7, .8], "calibration": [{"temperature": t} for t in [.5, .7, .8]],
                  "calibrated": True, "curriculum": {"frozen_backbone_and_final": True}}
        config = {"source_checkpoint": "fixture.pt", "lr": .001, "weight_decay": .0001,
                  "epochs": 2, "patience": 1, "ramp_epochs": 2, "loss_weights": [.2, .3, .5],
                  "calibration_steps": 5, "max_train_batches": None, "max_val_batches": None}
        return model, source, config

    def test_training_unfreezes_backbone_and_checks_selected_checkpoint(self):
        model, source, config = self.fixture()
        for p in model.parameters():
            p.requires_grad_(False)
        batches = [(torch.randn(8, 9, 128), torch.zeros(8, dtype=torch.long))]
        with tempfile.TemporaryDirectory() as root:
            result = train_scheduled(model, batches, batches, torch.device('cpu'), root, source, config)
            ckpt = torch.load(Path(root) / 'best.pt', weights_only=True)
            self.assertTrue(result['selection_constraint_verified'])
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            self.assertGreater(ckpt['epoch'], 0)
            self.assertGreater(int(ckpt['model_state']['stem.1.num_batches_tracked']),
                               int(source['model_state']['stem.1.num_batches_tracked']))
            self.assertNotIn('curriculum', ckpt)
            self.assertGreaterEqual(result['selected_validation']['final_accuracy'],
                                    result['baseline_validation']['final_accuracy'])

    def test_source_is_retained_when_every_candidate_is_ineligible(self):
        model, source, config = self.fixture()
        def metrics(accuracies, loss):
            return {'loss': loss, 'samples': 8,
                    **{f'{name}_accuracy': a for name, a in zip(EXIT_NAMES, accuracies)},
                    **{f'{name}_loss': loss for name in EXIT_NAMES}}
        baseline = metrics((.7, .8, .9), .4)
        calls = [baseline, metrics((.8, .9, .9), .2), metrics((.8, .9, .85), .1),
                 metrics((.8, .9, .9), .2), metrics((.91, .90, .92), .1), baseline]
        with tempfile.TemporaryDirectory() as root, patch('src.scheduled_training.run_epoch', side_effect=calls):
            result = train_scheduled(model, [], [], torch.device('cpu'), root, source, config)
            ckpt = torch.load(Path(root) / 'best.pt', weights_only=True)
            self.assertEqual(result['best_epoch'], 0)
            self.assertEqual(ckpt['temperatures'], source['temperatures'])
            for n, value in source['model_state'].items():
                self.assertTrue(torch.equal(ckpt['model_state'][n], value), n)


if __name__ == '__main__':
    unittest.main()
