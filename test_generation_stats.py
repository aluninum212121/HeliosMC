"""Regression tests for complete generation accounting and GUI integration."""
import os
os.environ["MPLBACKEND"] = "Agg"

import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from material import MATERIALS
from physics import estimate_k_full, run_generation_batch
from gui import NeutronSimGUI


class GenerationStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.material = next(iter(MATERIALS.values()))
        self.positions = np.zeros((125, 3))
        self.directions = np.tile([0., 0., 1.], (125, 1))

    def test_complete_counts_with_mixed_fates_and_small_sample(self):
        fates = np.array([0] * 50 + [1] * 35 + [2] * 25 + [-1] * 15)
        paths = [[p.copy()] for p in self.positions]
        transport_result = (self.positions, self.directions, fates,
                            list(self.positions[:50]), paths)
        with patch("physics.transport_batch", return_value=transport_result):
            result = run_generation_batch(self.positions, self.directions,
                                          self.material, 25, max_samples_per_gen=3)
        self.assertEqual(result[-1], {"fission": 50, "absorption": 35,
                                     "leak": 25, "unfinished": 15})
        self.assertEqual(sum(result[-1].values()), 125)
        self.assertEqual(len(result[5]), 3)

    def test_real_transport_step_limit_and_no_samples(self):
        for steps in (0, 1):
            with self.subTest(max_steps=steps):
                result = run_generation_batch(self.positions, self.directions,
                                              self.material, 25,
                                              max_samples_per_gen=0, max_steps=steps)
                self.assertEqual(sum(result[-1].values()), 125)
                self.assertEqual(result[5], [])
                if steps == 0:
                    self.assertEqual(result[-1]["unfinished"], 125)
        result = run_generation_batch(self.positions, self.directions,
                                      self.material, 25, max_steps=0)
        self.assertTrue(all(fate == "unfinished" for _, fate in result[5]))

    def test_empty_batch(self):
        result = run_generation_batch(self.positions[:0], self.directions[:0],
                                      self.material, 25)
        self.assertEqual(sum(result[-1].values()), 0)
        self.assertEqual(result[5], [])

    def test_real_generations_and_gui_histogram(self):
        result = estimate_k_full(n_gen=4, n_source=125, material=self.material)
        self.assertEqual(len(result["generation_stats"]), len(result["k_history"]))
        for stats, paths in zip(result["generation_stats"], result["path_frames"]):
            self.assertEqual(sum(stats["outcome_counts"].values()), stats["processed"])
            self.assertLessEqual(len(paths), 40)
        with patch("matplotlib.pyplot.show"):
            app = NeutronSimGUI()
        try:
            app.result = result
            for gen, stats in enumerate(result["generation_stats"]):
                app.current_gen = gen
                app._render_all()
                app.fig.canvas.draw()
                heights = [bar.get_height() for bar in app.ax_bar.patches]
                self.assertEqual(heights, list(stats["outcome_counts"].values()))
            # Regression: histogram must still work without visualization paths.
            app.result["path_frames"] = []
            app._draw_bar()
            self.assertEqual(sum(bar.get_height() for bar in app.ax_bar.patches),
                             result["generation_stats"][-1]["processed"])
        finally:
            plt.close(app.fig)


if __name__ == "__main__":
    unittest.main()
