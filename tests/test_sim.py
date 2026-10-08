import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from sim.__main__ import parse_battery_script
from sim.core import Metrics, Prediction, Simulator, infer
from sim.policy import BatteryController
from src.data import CLASS_NAMES, prepare_data
from src.metrics import classification_metrics
from src.model import AdaptiveHAR
from tests.fixtures import write_uci_fixture

ROOT = Path(__file__).resolve().parents[1]


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def simulator(self, size=8):
        torch.manual_seed(9)
        model = AdaptiveHAR().eval()
        dataset = TensorDataset(torch.randn(size, 9, 128), torch.arange(size) % 6)
        return Simulator(model, dataset, [1., 1., 1.], CLASS_NAMES)

    def test_exact_battery_boundaries_and_hysteresis(self):
        controller = BatteryController()
        states = [controller.update(b) for b in [80, 36, 35, 39.9, 40, 19.9, 20, 39, 40]]
        self.assertEqual(states, ["normal", "normal", "low-power", "low-power", "normal",
                                  "critical", "low-power", "low-power", "normal"])
        controller.update(0)
        controller.reset()
        self.assertEqual(controller.mode, "normal")
        for battery in [-1, 101, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                controller.update(battery)
        for arguments in [(35, 35, 20), (35, 40, 35), (35, 101, 20)]:
            with self.assertRaises(ValueError):
                BatteryController(*arguments)

    def test_full_and_critical_skip_unused_paths_and_have_exact_costs(self):
        simulator = self.simulator()
        model = simulator.model
        x = simulator.dataset[0][0].unsqueeze(0)
        with torch.inference_mode():
            expected = model(x)
        with patch.object(model.exit1, "forward", side_effect=AssertionError("unused exit")), \
             patch.object(model.exit2, "forward", side_effect=AssertionError("unused exit")):
            full = infer(model, x, simulator.temperatures, "full")
        self.assertEqual(full.prediction, int(expected[2].argmax(1)))
        self.assertEqual((full.exit, full.flops), (3, 156084))
        with patch.object(model.stage2, "forward", side_effect=AssertionError("unused stage")), \
             patch.object(model.stage3, "forward", side_effect=AssertionError("unused stage")):
            first = infer(model, x, simulator.temperatures, "critical")
        self.assertEqual(first.prediction, int(expected[0].argmax(1)))
        self.assertEqual((first.exit, first.flops), (1, 81456))
        with patch.object(model.stage3, "forward", side_effect=AssertionError("low-power depth cap")):
            low = infer(model, x, simulator.temperatures, "low-power")
        self.assertIn(low.exit, (1, 2))
        self.assertEqual(low.flops, {1: 81456, 2: 135748}[low.exit])
        self.assertFalse(any(module._forward_hooks for module in model.modules()))

    def test_paired_real_inference_and_deterministic_sampling_without_replacement(self):
        simulator = self.simulator(size=5)
        with patch("sim.core.infer", wraps=infer) as calls:
            records = [simulator.step(battery) for battery in [80, 30, 10, 20, 40]]
        self.assertEqual(calls.call_count, 15)
        self.assertEqual(len({r["dataset_index"] for r in records}), 5)
        self.assertTrue(all(r["pass"] == 1 for r in records))
        self.assertEqual(records[0]["outputs"]["battery"]["prediction"],
                         records[0]["outputs"]["normal"]["prediction"])
        self.assertEqual(records[0]["outputs"]["battery"]["exit"],
                         records[0]["outputs"]["normal"]["exit"])
        self.assertEqual(records[1]["mode"], "low-power")
        self.assertLessEqual(records[1]["outputs"]["battery"]["exit"], 2)
        self.assertEqual(records[2]["outputs"]["battery"]["exit"], 1)
        self.assertEqual(records[3]["mode"], "low-power")
        self.assertTrue(all(r["outputs"]["full"]["exit"] == 3 for r in records))
        self.assertEqual(simulator.step(80)["pass"], 2)
        simulator.reset()
        self.assertEqual(simulator.snapshot()["policies"]["full"]["samples"], 0)
        self.assertEqual([simulator.step(80)["dataset_index"] for _ in range(5)],
                         [r["dataset_index"] for r in records])

    def test_warmup_does_not_count_and_invalid_battery_does_not_consume_sample(self):
        simulator = self.simulator()
        simulator.warmup(1)
        self.assertEqual(simulator.records, [])
        with self.assertRaises(ValueError):
            simulator.step(float("nan"))
        self.assertEqual(simulator.cursor, 0)
        self.assertEqual(simulator.records, [])

    def test_mode_switches_never_reload_restart_or_mutate_model(self):
        simulator = self.simulator()
        identity = id(simulator.model)
        pointers = [p.data_ptr() for p in simulator.model.parameters()]
        original = {key: value.clone() for key, value in simulator.model.state_dict().items()}
        with patch("torch.load", side_effect=AssertionError("Mode switch must not reload checkpoint")):
            for battery in [80, 35, 19, 20, 39, 40]:
                simulator.step(battery)
                self.assertEqual(id(simulator.model), identity)
                self.assertEqual([p.data_ptr() for p in simulator.model.parameters()], pointers)
        for key, value in simulator.model.state_dict().items():
            torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    def test_plot_has_three_panels_and_metrics_survive_visible_history_cutoff(self):
        from sim.plotting import SimulationFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        simulator = self.simulator()
        for battery in [80, 30, 10, 40]:
            simulator.step(battery)
        plot = SimulationFigure(history=2)
        FigureCanvasAgg(plot.figure)
        plot.update(simulator)
        self.assertEqual(len(plot.axes), 3)
        self.assertEqual(list(plot.axes[0].lines[0].get_xdata()), [3, 4])
        self.assertEqual(simulator.snapshot()["policies"]["battery"]["samples"], 4)
        self.assertEqual([ax.yaxis.get_label_position() for ax in plot.accuracy_axes],
                         ["right", "right", "right"])
        plot.figure.canvas.draw()

    def test_metrics_match_existing_metrics_and_exports_are_safe(self):
        metrics = Metrics(6)
        labels, predictions = [0, 1, 2, 2], [0, 2, 2, 1]
        for label, prediction in zip(labels, predictions):
            metrics.add(label, Prediction(prediction, .9, 2, 135748, 3.))
        expected = classification_metrics(labels, predictions)
        actual = metrics.snapshot()
        self.assertEqual(actual["accuracy"], expected["accuracy"])
        self.assertAlmostEqual(actual["macro_f1"], expected["macro_f1"])
        self.assertEqual(actual["exit_counts"], [0, 4, 0])
        self.assertEqual(actual["total_flops"], 4 * 135748)
        simulator = self.simulator()
        for battery in [80, 30, 10]:
            simulator.step(battery)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "export"
            simulator.export(destination)
            report = json.loads((destination / "results.json").read_text())
            self.assertEqual(len(report["windows"]), 3)
            self.assertEqual(len((destination / "windows.csv").read_text().splitlines()), 10)
            self.assertEqual(report["policies"]["full"]["exit_counts"], [0, 0, 3])
            self.assertEqual([report["battery_modes"][mode]["samples"]
                              for mode in ("normal", "low-power", "critical")], [1, 1, 1])
            self.assertEqual(report["controller"]["critical_below"], 20)
            with self.assertRaises(FileExistsError):
                simulator.export(destination)

    def test_script_validation(self):
        self.assertEqual(parse_battery_script("0:80,2:10,4:40"), {0: 80., 2: 10., 4: 40.})
        for text in ["", "oops", "0:80,0:30", "2:80,1:30", "0:nan", "-1:80", "0:101"]:
            with self.assertRaises(ValueError):
                parse_battery_script(text)

    def test_headless_cli_from_fixture_produces_three_stacked_graphs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_uci_fixture(tmp / "data", windows_per_subject=6)
            bundle = prepare_data(tmp / "data", val_subjects=[5])
            model = AdaptiveHAR()
            checkpoint = tmp / "fixture.pt"
            torch.save({"model_config": model.config, "model_state": model.state_dict(),
                        "data_metadata": bundle.metadata, "calibrated": True,
                        "temperatures": [1., 1., 1.]}, checkpoint)
            command = [sys.executable, "-m", "sim", "--headless", "--checkpoint", str(checkpoint),
                       "--data-dir", str(tmp / "data"), "--steps", "6",
                       "--battery-script", "0:80,2:30,4:10", "--output-dir", str(tmp / "out")]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((tmp / "out/results.json").read_text())
            self.assertTrue(report["provenance"]["synthetic_data"])
            self.assertEqual(report["policies"]["full"]["samples"], 6)
            self.assertEqual(report["battery_modes"]["critical"]["exit_counts"], [2, 0, 0])
            self.assertTrue((tmp / "out/live_comparison.png").exists())
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
