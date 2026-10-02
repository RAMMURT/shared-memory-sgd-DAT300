"""Run with: python -m unittest discover -s tests -v"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment_viewer import Experiment, export_all, figures


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "run.json"

    def load(self, **fields):
        data = {"meta": {"dispatcher": "async", "num_threads": 16, "dataset": "CIFAR10"}}
        data.update(fields)
        self.path.write_text(json.dumps(data))
        return Experiment(self.path)

    def test_weighted_statistics_and_first_best_epoch(self):
        run = self.load(tau_dist=[1, 2, 1], epoch_accuracy=[.2, .8, .8], epoch_mstimes=[1000, 2000, 3000], end_accuracy=-1)
        summary = dict(run.summary())
        self.assertEqual(run.best_index, 1)
        self.assertEqual(summary["Best accuracy"], "80.000% at epoch 2")
        self.assertEqual(summary["Time to best accuracy"], "2.000 s")
        self.assertEqual(summary["Last recorded accuracy"], "80.000%")
        self.assertEqual(summary["Mean τ"], "1.000")
        self.assertEqual(summary["Median τ"], "1")
        self.assertEqual(summary["p95 τ"], "2")
        self.assertEqual(summary["Standard deviation τ"], "0.707")

    def test_empty_run_has_no_fake_best_or_tau(self):
        run = self.load(tau_dist=[0] * 512, epoch_accuracy=[], epoch_mstimes=[])
        summary = dict(run.summary())
        self.assertIsNone(run.best_index)
        self.assertEqual(summary["Best accuracy"], "No recorded data")
        self.assertEqual(summary["τ statistics"], "No recorded data")
        self.assertEqual(len(figures(run)), 4)

    def test_nulls_and_mismatched_time_arrays(self):
        run = self.load(epoch_accuracy=[None, .5, .9], epoch_mstimes=[1000])
        self.assertEqual(run.best_index, 2)
        self.assertEqual(dict(run.summary())["Time to best accuracy"], "Unavailable")
        self.assertTrue(any("lengths differ" in note for note in run.notes))
        figures(run)  # No indexing beyond the timestamp array.

    def test_loss_indices_and_zero_bins_between_peaks(self):
        run = self.load(epoch_loss=[3, 2], alt_epoch_loss=[4, 3, 2], tau_dist=[10, 0, 0, 10, 0])
        plots = dict(figures(run))
        np.testing.assert_equal(plots["loss_epoch"].axes[0].lines[0].get_xdata(), [1, 2])
        np.testing.assert_equal(plots["loss_epoch"].axes[0].lines[1].get_xdata(), [0, 1, 2])
        bars = plots["tau_histogram"].axes[0].patches
        self.assertEqual([p.get_height() for p in bars], [10, 0, 0, 10])

    def test_invalid_counts_rejected(self):
        for counts in ([1, -1], [1.5], [None], [float("nan")]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                self.load(tau_dist=counts)

    def test_invalid_accuracy_and_clock_excluded(self):
        run = self.load(epoch_accuracy=[-1, .4, 2], epoch_mstimes=[1000, 500, 2000])
        self.assertEqual(run.best_index, 1)
        self.assertEqual(dict(run.summary())["Recorded duration"], "Unavailable")

    def test_optional_measurements_export_in_all_formats(self):
        run = self.load(epoch_accuracy=[.2, .7], epoch_mstimes=[1000, 2000],
                        tau_dist=[1, 2], epoch_tau_dist=[[0, 0], [1, 2]],
                        async_period_values=[8, 4], async_period_mstimes=[0, 1000],
                        steptimes=[[0, 1000000, 0, 1], [2000000, 4000000, 1, 0]])
        for fmt in ("png", "svg", "pdf"):
            output = export_all(run, Path(self.temp.name) / fmt, fmt)
            self.assertTrue((output / f"tau_histogram.{fmt}").stat().st_size > 100)
            self.assertTrue((output / f"tau_by_epoch.{fmt}").exists())
            self.assertTrue((output / f"step_duration.{fmt}").exists())
            self.assertEqual(json.loads((output / "summary.json").read_text())["meta"], run.meta)


if __name__ == "__main__":
    unittest.main()
