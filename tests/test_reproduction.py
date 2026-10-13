import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from scripts.reproduce_model import model_state_digest


PROJECT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_recipe_identifies_the_selected_model(self):
        recipe = json.loads((PROJECT / "models/adaptive_har.recipe.json").read_text())
        checkpoint = torch.load(PROJECT / "models/adaptive_har.pt", map_location="cpu", weights_only=True)
        self.assertEqual(model_state_digest(checkpoint["model_state"]), recipe["reference"]["model_state_sha256"])
        self.assertEqual(checkpoint["temperatures"], recipe["reference"]["temperatures"])
        self.assertEqual(checkpoint["epoch"], recipe["reference"]["epoch"])
        self.assertEqual(recipe["training"]["val_subjects"], checkpoint["data_metadata"]["validation_subjects"])
        for key, value in recipe["training"].items():
            if key != "val_subjects":
                self.assertEqual(value, checkpoint["training_config"][key], key)

    def test_fingerprint_is_stable_and_detects_changed_weights_and_buffers(self):
        state = {"weight": torch.tensor([[1., 2.]]), "bn.num_batches_tracked": torch.tensor(5)}
        expected = model_state_digest(state)
        reordered = {name: value.clone() for name, value in reversed(list(state.items()))}
        self.assertEqual(expected, model_state_digest(reordered))
        for name in state:
            changed = {key: value.clone() for key, value in state.items()}
            changed[name] += 1
            self.assertNotEqual(expected, model_state_digest(changed))
        reshaped = {**state, "weight": state["weight"].reshape(2, 1)}
        self.assertNotEqual(expected, model_state_digest(reshaped))
        renamed = {"other_weight": state["weight"], "bn.num_batches_tracked": state["bn.num_batches_tracked"]}
        self.assertNotEqual(expected, model_state_digest(renamed))

    def test_existing_checkpoint_is_preserved_without_starting_training(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.write_bytes(b"existing checkpoint")
            result = subprocess.run([sys.executable, str(PROJECT / "scripts/reproduce_model.py"),
                                     "--output-dir", directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Output checkpoint already exists", result.stderr)
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")


if __name__ == "__main__":
    unittest.main()
