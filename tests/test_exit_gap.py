import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.metrics import EXIT_NAMES
from src.model import AdaptiveHAR
from src.training import exit_gap_score, train_model


def metrics(accuracies, loss):
    return {"loss": loss, "samples": 8,
            **{f"{name}_accuracy": value for name, value in zip(EXIT_NAMES, accuracies)},
            **{f"{name}_loss": loss for name in EXIT_NAMES}}


class ExitGapTests(unittest.TestCase):
    def test_both_gaps_are_required_including_at_the_exact_boundary(self):
        self.assertTrue(exit_gap_score(metrics((.80, .81, .82), .2), .01)[0])
        self.assertFalse(exit_gap_score(metrics((.80, .81, .8199), .2), .01)[0])
        self.assertFalse(exit_gap_score(metrics((.805, .81, .90), .2), .01)[0])
        self.assertFalse(exit_gap_score(metrics((.80, .90, .89), .2), .01)[0])
        eligible = exit_gap_score(metrics((.80, .85, .90), .3), .01)
        ineligible = exit_gap_score(metrics((.94, .95, .955), .1), .01)
        self.assertGreater(eligible, ineligible)
        self.assertGreater(exit_gap_score(metrics((.80, .85, .91), .4), .01), eligible)

    def test_saved_checkpoint_and_summary_follow_validation_gap_selection(self):
        cases = [
            ([(.80, .85, .90), (.88, .91, .915), (.81, .86, .92)], 3, True),
            ([(.80, .85, .90), (.88, .91, .93), (.90, .91, .945)], 2, True),
            ([(.90, .91, .915), (.91, .92, .925), (.90, .94, .935)], 3, False),
        ]
        config = {"lr": .001, "weight_decay": .0001, "epochs": 3, "patience": 3,
                  "min_delta": 0., "loss_weights": [.05, .10, .85],
                  "max_train_batches": None, "max_val_batches": None,
                  "calibration_steps": 5, "checkpoint_metric": "exit-gap", "min_exit_gap": .02}
        torch.set_num_threads(2)
        logits, labels = (torch.zeros(8, 6),) * 3, torch.arange(8) % 6
        for accuracies, expected_epoch, target_met in cases:
            calls = [row for values in accuracies
                     for row in (metrics((.7, .8, .9), .4), metrics(values, .2))]
            with self.subTest(accuracies=accuracies), tempfile.TemporaryDirectory() as root, \
                 patch("src.training.run_epoch", side_effect=calls), \
                 patch("src.training.collect_logits", return_value=(logits, labels)), \
                 patch("src.training.plot_history"):
                result = train_model(AdaptiveHAR(), [], [], torch.device("cpu"), root,
                                     {"source": "synthetic_test_fixture"}, config)
                checkpoint = torch.load(Path(root) / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["epoch"], expected_epoch)
                target = result["exit_gap_target"]
                self.assertEqual(target["validation_target_met"], target_met)
                self.assertEqual(target["minimum_gap_percentage_points"], 2.)
                self.assertEqual(target["selected_validation"]["final_accuracy"],
                                 accuracies[expected_epoch - 1][2])


if __name__ == "__main__":
    unittest.main()
