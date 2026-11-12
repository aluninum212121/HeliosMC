"""Regression tests for complete generation accounting and GUI integration."""
import os
os.environ["MPLBACKEND"] = "Agg"

import unittest
from copy import deepcopy
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from material import (MATERIALS, MaterialDataset, MaterialQuantity,
                      REFERENCE_POINT_PROFILE, REFERENCE_CONTROL_PROFILE,
                      REFERENCE_MAXWELL_PROFILE, get_reference_profiles)
from physics import (estimate_k_full, run_generation_batch, load_slab_experiment,
                     run_slab_experiment, create_slab_configuration,
                     export_slab_experiment)
from gui import NeutronSimGUI
from simulation_worker import ExecutionState


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


@pytest.fixture
def experiment_gui():
    """Exercise the real controller and Matplotlib rendering without a Tk root."""
    app = NeutronSimGUI(show=False)
    try:
        yield app
    finally:
        app._on_close(None)
        for figure in app.figures.values():
            plt.close(figure)


def set_slab_inputs(app, *, reference=False, material="Вода (H₂O)",
                    profile=REFERENCE_POINT_PROFILE):
    app._set("scenario", "Транспорт через слой")
    app._on_scenario_change()
    for name, value in {
        "material_mode": "Reference" if reference else "Учебная среда",
        "material": material, "profile": profile,
        "width": "1.0", "histories": "800", "seed": "6101",
        "max_collisions": "100", "samples": "4",
        "absorption": "0.5", "scattering": "0",
        "density": "", "density_sample": "", "density_method": "",
    }.items():
        app._set(name, value)


def test_gui_slab_worker_export_open_preserves_seed_and_uncertainty(experiment_gui, tmp_path):
    app = experiment_gui
    set_slab_inputs(app)
    app._on_run(None)
    assert app.worker.thread is not None
    app.worker.thread.join(timeout=60)
    assert not app.worker.thread.is_alive()
    app._poll_worker()
    assert app.execution_state == ExecutionState.COMPLETED
    assert app.result_scenario == "slab"
    assert app.result == app.experiment["result"]
    archived = deepcopy(app.experiment)
    configuration = archived["configuration"]
    assert configuration["material_mode"] == "legacy"
    assert configuration["parameters"]["seed"] == 6101
    assert archived["result"]["processed_histories"] == 800
    estimate = archived["result"]["estimates"]["escaped_right"]
    assert estimate["standard_error"] > 0
    assert len(estimate["confidence_interval"]) == 2

    destination = tmp_path / "gui-experiment"
    app.save_experiment(destination)
    assert {"experiment.json", "results.csv", "arrays.npz"} <= {
        path.name for path in destination.iterdir()}
    assert load_slab_experiment(destination) == archived
    app.open_experiment(destination)
    assert app.experiment == archived
    assert app.result == archived["result"]
    assert app.result_scenario == "slab"
    assert int(app._get("seed")) == 6101
    replay = run_slab_experiment(app.experiment["configuration"])
    assert replay["result"]["outcome_counts"] == archived["result"]["outcome_counts"]
    assert replay["result"]["estimates"] == archived["result"]["estimates"]


def test_gui_loaded_multilayer_archive_replays_complete_configuration(experiment_gui, tmp_path):
    def medium(identifier, absorption, scattering):
        return MaterialDataset(identifier, identifier, "synthetic",
            "Invented GUI replay fixture; not measured material properties", "1",
            ("Software verification only",),
            (MaterialQuantity("absorption_rate", "1/cm", absorption),
             MaterialQuantity("scattering_rate", "1/cm", scattering)))

    configuration = create_slab_configuration("gui-loaded-multilayer",
        [medium("fixture:first-layer", .2, .3), medium("fixture:second-layer", .7, .1)],
        [0., .8, 2.], n_histories=1200, seed=12987, source_x=.2,
        source_mu=.75, confidence=.9, backend="python", batch_size=37,
        max_collisions=100, max_samples=0)
    frozen = deepcopy(configuration)
    archived = run_slab_experiment(configuration)
    destination = tmp_path / "multilayer-archive"
    export_slab_experiment(archived, destination)
    app = experiment_gui
    app.open_experiment(destination)
    assert app.loaded_configuration == frozen
    assert app.result_config == frozen
    assert float(app._get("absorption")) == .2
    assert float(app._get("scattering")) == .3

    # The compact editor represents the first layer. Replay must use every
    # archived layer and source/uncertainty option, rather than reconstructing
    # a one-layer run from the visible fields.
    app._on_run(None)
    assert app.worker.thread is not None
    app.worker.thread.join(timeout=60)
    assert not app.worker.thread.is_alive()
    app._poll_worker()
    assert app.execution_state == ExecutionState.COMPLETED
    assert app.result_config == frozen
    assert app.experiment["configuration"] == frozen
    assert app.loaded_configuration == frozen
    assert configuration == frozen
    assert app.result["outcome_counts"] == archived["result"]["outcome_counts"]
    assert app.result["estimates"] == archived["result"]["estimates"]
    assert app.result["run_info"]["source_x_cm"] == .2
    assert app.result["run_info"]["source_mu"] == .75
    assert app.result["run_info"]["confidence"] == .9
    assert app.result["run_info"]["backend"] == "python_numpy"


@pytest.mark.parametrize("material,profile", [
    ("Вода (H₂O)", REFERENCE_POINT_PROFILE),
    ("Графит", REFERENCE_POINT_PROFILE),
    ("Графит", REFERENCE_CONTROL_PROFILE),
    ("Графит", REFERENCE_MAXWELL_PROFILE),
])
def test_gui_reference_selection_keeps_existing_profile_contract(experiment_gui, material, profile):
    app = experiment_gui
    set_slab_inputs(app, reference=True, material=material, profile=profile)
    if material == "Графит":
        # Explicit fixture lineage, never claimed to be a measured graphite grade.
        app._set("density", "1.6")
        app._set("density_sample", "Invented graphite sample for GUI testing")
        app._set("density_method", "Caller input for software verification only")
    configuration = app._build_slab_configuration()
    expected = get_reference_profiles()[profile]
    assert configuration["material_mode"] == "reference"
    assert configuration["reference_profile"] == expected
    assert configuration["material_model_version"] == expected["approximation"]
    assert configuration["parameters"]["seed"] == 6101
    assert configuration["parameters"]["edges_cm"] == [0., 1.]
    assert configuration["layers"][0]["data_kind"] == "reference"
    if profile == REFERENCE_POINT_PROFILE:
        assert expected["spectrum_kind"] == "point"
        assert expected["energy_bounds_ev"] == [.0253, .0253]
    else:
        assert expected["source_library"] == "JENDL-4.0"


@pytest.mark.parametrize("density,sample,method", [
    ("", "", ""),
    ("1.6", "", ""),
    ("1.6", "Invented sample", ""),
])
def test_gui_reference_graphite_rejects_missing_density_lineage(
        experiment_gui, density, sample, method):
    app = experiment_gui
    set_slab_inputs(app, reference=True, material="Графит")
    for name, value in (("density", density), ("density_sample", sample),
                        ("density_method", method)):
        app._set(name, value)
    with pytest.raises(ValueError):
        app._build_slab_configuration()
    app._on_run(None)
    assert app.execution_state == ExecutionState.FAILED
    assert app.worker.thread is None
    assert app.result is None


@pytest.mark.parametrize("profile", [REFERENCE_MAXWELL_PROFILE, "unknown-profile"])
def test_gui_reference_water_cannot_fall_back_to_a_different_profile(experiment_gui, profile):
    app = experiment_gui
    set_slab_inputs(app, reference=True, profile=profile)
    with pytest.raises(ValueError):
        app._build_slab_configuration()


def test_gui_archived_slab_does_not_manufacture_spatial_results(experiment_gui):
    app = experiment_gui
    set_slab_inputs(app)
    app.experiment = run_slab_experiment(app._build_slab_configuration())
    app.result = app.experiment["result"]
    app.result_scenario = "slab"
    before = deepcopy(app.experiment)
    app._draw_heatmap()
    app._draw_trajectories()
    assert not app.ax_heat.collections and not app.ax_heat.images
    assert not app.ax_main.collections and not app.ax_main.images
    assert not app.ax_main.lines
    assert app.ax_heat.texts and app.ax_main.texts
    assert app.experiment == before


@pytest.mark.parametrize("frame_representation", ["empty", "per_generation"])
def test_gui_no_path_sample_refuses_animation_and_explains_absence(
        experiment_gui, frame_representation):
    app = experiment_gui
    app.result = estimate_k_full(n_gen=2, n_source=60, seed=801,
                                 max_samples_per_gen=0)
    if frame_representation == "empty":
        app.result["path_frames"] = []
    app.result_scenario = "generations"
    app.result_radius = 25.
    app._draw_trajectories()
    assert "Траектории не сохранены" in "\n".join(
        artist.get_text() for artist in app.ax_main.texts)
    assert not app.ax_main.lines and not app.ax_main.collections
    with patch("gui.animation.FuncAnimation") as animation_factory:
        app._on_anim(None)
    animation_factory.assert_not_called()
    assert not app.anim_running
    assert app.anim_obj is None
    assert "Нет сохранённых траекторий" in app.status_text


def test_gui_heatmap_resolution_and_plot_export_preserve_numerics(experiment_gui, tmp_path):
    app = experiment_gui
    app.result = estimate_k_full(n_gen=3, n_source=80, seed=1402,
                                 max_samples_per_gen=4)
    app.result_scenario = "generations"
    app.result_radius = 25.
    app._render_all()
    history = list(app.result["k_history"])
    statistics = deepcopy(app.result["generation_stats"])
    fission_points = [frame.copy() for frame in app.result["fiss_frames"]]
    figure_count = len(plt.get_fignums())
    axes_count = {name: len(figure.axes) for name, figure in app.figures.items()}
    for resolution in (20, 48, 80, 20):
        app.heat_bins = resolution
        app._draw_heatmap()
        rasters = list(app.ax_heat.collections) + list(app.ax_heat.images)
        assert len(rasters) == 1
        counts = np.ma.asarray(rasters[0].get_array())
        assert counts.size == resolution ** 2
        assert int(np.ma.filled(counts, 0).sum()) == sum(map(len, fission_points))
        assert "см" in app.ax_heat.get_xlabel()
        assert "см" in app.ax_heat.get_ylabel()
        assert len(plt.get_fignums()) == figure_count
        assert {name: len(figure.axes) for name, figure in app.figures.items()} == axes_count
    image = tmp_path / "source-reproduction.png"
    app.save_plot(image, tab="graphs")
    assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert image.stat().st_size > 1000
    assert app.result["k_history"] == history
    assert app.result["generation_stats"] == statistics
    for before, after in zip(fission_points, app.result["fiss_frames"]):
        np.testing.assert_array_equal(after, before)


if __name__ == "__main__":
    unittest.main()
