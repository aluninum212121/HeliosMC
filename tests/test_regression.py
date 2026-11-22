"""Completion reporting and seeded compatibility regressions."""
import os
os.environ['MPLBACKEND'] = 'Agg'
from threading import Event
from unittest.mock import patch
from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest

import physics
from material import (Material, TransportData, resolve_transport_data, MaterialDataset,
                      MaterialQuantity, MATERIALS, get_builtin_dataset)
from gui import NeutronSimGUI
from simulation_worker import SimulationWorker, ExecutionState


def material(f=0., a=0., s=1., nu=2.43):
    return Material('synthetic', f, a, s, nu, 1., 1., 'red')


@pytest.mark.parametrize('mat,steps,limited,completed', [
    (material(a=1., s=0.), 2, 0, 100),
    (material(), 0, 100, 0),
    (material(a=.5, s=.5), 1, None, None),
])
def test_completion_metadata(mat, steps, limited, completed):
    r = physics.estimate_k_full(n_gen=3, n_source=100, material=mat,
                               radius=1e9, max_steps=steps, seed=17)
    counts = r['generation_stats'][0]['outcome_counts']
    if limited is None:
        assert 0 < counts['unfinished'] < 100
        limited = counts['unfinished']
        completed = 100 - limited
    assert r['step_limited_histories'] == limited
    assert r['completed_histories'] == completed
    assert r['histories_complete'] == (limited == 0)
    assert r['execution_status'] == 'completed'
    assert r['termination_reason'] == 'no_secondary_source'
    s = r['generation_stats'][0]
    assert s['completed'] + s['step_limited'] == s['processed'] == sum(counts.values())
    assert s['histories_complete'] == r['histories_complete']


@pytest.mark.parametrize('config,reason', [({'n_source': 0}, 'empty_source'),
                                         ({'n_gen': 0}, 'no_generations')])
def test_no_histories(config, reason):
    r = physics.estimate_k_full(**config)
    assert r['termination_reason'] == reason
    assert r['histories_complete']
    assert r['completed_histories'] == r['step_limited_histories'] == 0
    assert r['k_history'] == []


def test_estimator_compatibility_with_unfinished_denominator():
    counts = dict(fission=2, absorption=1, leak=1, unfinished=6)
    generation = (np.zeros((4, 3)), np.tile([0., 0., 1.], (4, 1)),
                  2, 1, 1, [], np.zeros((2, 3)), counts)
    with patch('physics.run_generation_batch', return_value=generation):
        r = physics.estimate_k_full(n_gen=1, n_source=10, seed=1)
    assert r['k_history'] == [.4]  # 4 / 10, never 4 / 4
    assert r['completed_histories'] == 4
    assert r['step_limited_histories'] == 6


@pytest.mark.parametrize('steps,source,expected', [(0, 100, 'НЕПОЛНЫЙ РЕЗУЛЬТАТ'),
                                                 (2, 100, 'СРЕДНЕЕ k < 0,99'),
                                                 (2, 0, 'НЕТ ОЦЕНКИ')])
def test_gui_completeness_classification(steps, source, expected):
    app = NeutronSimGUI(show=False)
    try:
        app.result = physics.estimate_k_full(n_gen=1, n_source=source, max_steps=steps,
                                            material=material(a=1., s=0.), seed=1)
        app._render_all()
        text = '\n'.join(t.get_text() for t in app.ax_stat.texts)
        assert expected in text
        if steps == 0:
            assert 'Оценка может быть неполной' in text
            assert 'СРЕДНЕЕ k < 0,99' not in text
            assert 'Лимит шагов: 100/100 (100.0%)' in text
        marked = any(artist.get_label() == 'Лимит шагов'
                     for artist in app.ax_k.collections)
        assert marked == (steps == 0)
        app.fig.canvas.draw()
        assert all(t.get_position()[1] > 0 for t in app.ax_stat.texts)
    finally:
        app._on_close(None)
        plt.close(app.fig)


@pytest.mark.parametrize('history,expected', [
    ([1.4, .8, .9, 29 / 26], 'СРЕДНЕЕ k < 0,99'),
    ([.2, 1.1, 1.2, 1.3], 'СРЕДНЕЕ k > 1,01'),
    ([.2, .9, 1., 1.1], 'СРЕДНЕЕ k ≈ 1'),
])
def test_gui_separates_generation_coefficient_from_active_mean(history, expected):
    # Display fixtures, not physical measurements. The final point is above
    # one in every case, whereas the selected mean can be below/above/near one.
    result = dict(k_history=history, k_mean=float(np.mean(history[1:])),
                  k_std=float(np.std(history[1:])), path_frames=[], fiss_frames=[],
                  estimator=dict(active_generation_start=1),
                  generation_stats=[dict(processed=n, outcome_counts=dict(
                      fission=0, absorption=n, leak=0, unfinished=0))
                      for n in (1400, 100, 19, 26)])
    app = NeutronSimGUI(show=False)
    try:
        app.result = result
        app.current_gen = 3
        app.result_config = dict(material=next(iter(MATERIALS.values())), n_source=1400)
        app._render_all()
        text = '\n'.join(t.get_text() for t in app.ax_stat.texts)
        assert f"Среднее k = {result['k_mean']:.4f}" in text
        assert 'По поколениям 2–4' in text
        assert f'k поколения 4 = {history[3]:.4f}' in text
        assert 'Нейтронов в поколении: 26' in text
        assert 'Начальный источник: 1400' in text
        assert expected in text
        assert 'КРИТИЧНЫЙ' not in text
        assert 'обогащение' not in text
        np.testing.assert_array_equal(app.ax_k.lines[0].get_ydata(), history)
        mean_line = next(line for line in app.ax_k.lines
                         if line.get_label().startswith('Среднее k ='))
        np.testing.assert_array_equal(mean_line.get_ydata(), [result['k_mean']] * 2)
        assert f'k поколения = {history[3]:.4f}' in app.ax_main.get_title(loc='left')

        # Both button navigation and animation update the selected population
        # and coefficient without changing the overall mean or selected range.
        app._step_gen(-1)
        text = '\n'.join(t.get_text() for t in app.ax_stat.texts)
        assert f'k поколения 3 = {history[2]:.4f}' in text
        assert 'Нейтронов в поколении: 19' in text
        assert f"Среднее k = {result['k_mean']:.4f}" in text
        # Playback requires a genuinely available stored path sample.
        result['path_frames'] = [[(np.zeros((2, 3)), 'absorption')]]
        with patch('gui.animation.FuncAnimation') as animate:
            app._on_anim(None)
            update = animate.call_args.args[1]
            update(0)
        text = '\n'.join(t.get_text() for t in app.ax_stat.texts)
        assert f'k поколения 1 = {history[0]:.4f}' in text
        assert f"Среднее k = {result['k_mean']:.4f}" in text
        app.fig.canvas.draw()
        assert all(t.get_position()[1] > 0 for t in app.ax_stat.texts)
    finally:
        app._on_close(None)
        plt.close(app.fig)


def test_gui_short_run_shows_actual_averaging_range():
    app = NeutronSimGUI(show=False)
    try:
        # Fewer generations than burn-in: the unchanged estimator uses all
        # available generations. The GUI must not claim they were excluded.
        app.result = physics.estimate_k_full(n_gen=1, n_source=10, n_burnin=8,
                                            material=material(a=1., s=0.), seed=1)
        app._render_all()
        text = '\n'.join(t.get_text() for t in app.ax_stat.texts)
        assert 'По поколениям 1–1' in text
        assert 'k поколения 1 = 0.0000' in text
        assert not app.ax_k.patches  # no discarded-generation shading
    finally:
        app._on_close(None)
        plt.close(app.fig)


def test_cancellation_after_incomplete_generation():
    event = Event()
    with pytest.raises(physics.SimulationCancelled):
        physics.estimate_k_full(n_gen=1, max_steps=0, cancel_event=event,
                               callback=lambda *a: event.set())


def test_warning_survives_later_complete_generations():
    app = NeutronSimGUI(show=False)
    try:
        r = physics.estimate_k_full(n_gen=1, max_steps=0, seed=1)
        # Earlier unresolved histories also affect later source generations.
        r['generation_stats'].append(dict(processed=1,
            outcome_counts=dict(fission=1, absorption=0, leak=0, unfinished=0)))
        r['k_history'].append(1.)
        app.result = r
        app.current_gen = 1
        app._update_status_full()
        assert 'НЕПОЛНЫЙ РЕЗУЛЬТАТ' in [t.get_text() for t in app.ax_stat.texts]
        app._draw_k_chart()
        markers = [artist for artist in app.ax_k.collections
                   if artist.get_label() == 'Лимит шагов']
        assert len(markers) == 1
        assert markers[0].get_offsets()[:, 0].tolist() == [1]
    finally:
        app._on_close(None)
        plt.close(app.fig)


@pytest.fixture(scope='module')
def baseline():
    module = types.ModuleType('upgrade_test_baseline')
    sys.modules[module.__name__] = module
    source = Path('benchmarks/fixtures/baseline_physics.txt').read_bytes()
    exec(compile(source, '<pre-upgrade working tree>', 'exec'), module.__dict__)
    return module


@pytest.mark.parametrize('bitgen', [np.random.PCG64, np.random.PCG64DXSM,
                                   np.random.MT19937, np.random.Philox, np.random.SFC64])
@pytest.mark.parametrize('n', [0, 1, 17, 150, 2000])
def test_secondary_exact_reference_and_rng_continuation(baseline, bitgen, n):
    # Unique positions and mixed fates expose site ordering and truncation bugs.
    p = np.arange(n * 3, dtype=float).reshape(n, 3)
    d = np.tile([0., 0., 1.], (n, 1))
    fates = np.resize(np.array([0, 1, -1, 2, 0], dtype=np.int32), n)
    transport = (p, d, fates, list(p[fates == 0]), [])
    a = np.random.Generator(bitgen(2026))
    b = np.random.Generator(bitgen(2026))
    with patch.object(baseline, 'transport_batch', return_value=transport), \
         patch.object(physics, 'transport_batch', return_value=transport):
        x = baseline.run_generation_batch(p, d, material(), 1e9, 0, rng=a)
        y = physics.run_generation_batch(p, d, material(), 1e9, 0, rng=b)
    for i in (0, 1, 6):
        np.testing.assert_array_equal(x[i], y[i])
        assert x[i].shape == y[i].shape and x[i].dtype == y[i].dtype
    assert x[-1] == y[-1]
    assert x[2:6] == y[2:6]
    np.testing.assert_array_equal(a.bytes(101), b.bytes(101))


def test_full_seeded_reference_and_cap(baseline):
    mat = material(f=.7, a=.1, s=.2)
    config = dict(n_gen=4, n_source=150, radius=1e9, material=mat, seed=2026,
                  max_steps=8, max_samples_per_gen=7, n_burnin=1)
    x, y = baseline.estimate_k_full(**config), physics.estimate_k_full(**config)
    # Population control must not clip the reported ratio to one.
    assert y['k_mean'] > 1
    for key in ('k_history', 'k_mean', 'k_std'):
        assert x[key] == y[key]
    for a, b in zip(x['generation_stats'], y['generation_stats']):
        assert a['processed'] == b['processed'] <= 150
        assert a['outcome_counts'] == b['outcome_counts']
    for a, b in zip(x['fiss_frames'], y['fiss_frames']):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(x['path_frames'], y['path_frames']):
        for (pa, fa), (pb, fb) in zip(a, b):
            np.testing.assert_array_equal(pa, pb)
            assert fa == fb


def test_noncontiguous_inputs_and_custom_rng(baseline):
    p = np.zeros((20, 6))[:, ::2]
    d = np.tile([0., 0., 1., 0., 0., 1.], (20, 1))[:, ::2]
    # Selected columns are [0,1,0], hence still unit vectors.
    class ScalarRNG:
        def __init__(self):
            self.rng = np.random.default_rng(21)
        def standard_normal(self, shape):
            return self.rng.standard_normal(shape)
        def integers(self, low, high):
            return self.rng.integers(low, high)
    mat = material(f=1., s=0.)
    x = baseline.run_generation_batch(p, d, mat, 1e9, rng=ScalarRNG(), seed=2)
    y = physics.run_generation_batch(p, d, mat, 1e9, rng=ScalarRNG(), seed=2)
    np.testing.assert_array_equal(x[0], y[0])
    np.testing.assert_array_equal(x[1], y[1])
    assert not p.any()


@pytest.mark.parametrize('budget', [-1, True, 1.5])
def test_invalid_memory_budget(budget):
    with pytest.raises(ValueError, match='max_memory_bytes'):
        physics.estimate_k_full(max_memory_bytes=budget)


def test_memory_guards_before_allocation_and_secondary_expansion():
    with patch('physics.np.zeros', side_effect=AssertionError('allocated source')):
        with pytest.raises(MemoryError, match='storage estimate'):
            physics.estimate_k_full(n_source=10**10)
    with pytest.raises(MemoryError, match='storage estimate'):
        physics.estimate_k_full(n_source=20, n_gen=1, max_samples_per_gen=0,
                               material=material(f=1., s=0., nu=1e9), radius=1e9,
                               max_memory_bytes=100000, seed=2)
    x = physics.estimate_k_full(n_gen=2, seed=2, max_memory_bytes=None)
    y = physics.estimate_k_full(n_gen=2, seed=2)
    assert x['k_history'] == y['k_history']


def test_retained_frames_count_toward_budget():
    # One fission generation fits, but retained events consume the next budget.
    with pytest.raises(MemoryError, match='storage estimate'):
        physics.estimate_k_full(n_gen=20, n_source=100, max_samples_per_gen=0,
            material=material(f=1., s=0., nu=1.), radius=1e9, seed=1,
            max_memory_bytes=65000)


def test_worker_distinguishes_incomplete_execution_from_failure():
    worker = SimulationWorker()
    worker.start(physics.estimate_k_full, n_gen=1, max_steps=0, seed=1)
    worker.thread.join(timeout=30)
    state, result = worker.poll()
    assert state == ExecutionState.COMPLETED
    assert result['execution_status'] == 'completed' and not result['histories_complete']
    worker.start(physics.estimate_k_full, n_source=10**10)
    worker.thread.join(timeout=30)
    state, error = worker.poll()
    assert state == ExecutionState.FAILED and 'MemoryError' in error


@pytest.mark.parametrize('dtype', [np.float32, np.int32])
def test_generation_validation_and_input_ownership(dtype):
    p = np.zeros((12, 3), dtype=dtype)
    d = np.tile(np.array([0, 0, 1], dtype=dtype), (12, 1))
    old_p, old_d = p.copy(), d.copy()
    result = physics.run_generation_batch(p, d, material(f=1., s=0.), 1e9, seed=1)
    assert result[0].dtype == result[1].dtype == np.float64
    np.testing.assert_array_equal(p, old_p)
    np.testing.assert_array_equal(d, old_d)
    with pytest.raises(ValueError, match='unit vectors'):
        physics.run_generation_batch(p, np.zeros_like(d), material(), 1e9)


def test_transport_event_order_and_owning_public_arrays(baseline):
    p = np.arange(300, dtype=float).reshape(100, 3) / 100
    d = np.tile([0., 0., 1.], (100, 1))
    args = (p, d, 1., .3, .2, .5, 2.43, 1e9)
    a = baseline.transport_batch(*args, max_steps=8, max_samples=9, seed=2026)
    b = physics.transport_batch(*args, max_steps=8, max_samples=9, seed=2026)
    assert type(a[3]) is type(b[3]) is list
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    for x, y in zip(a[4], b[4]):
        np.testing.assert_array_equal(x, y)
    saved = [row.copy() for row in b[3]]
    b[0][:] = 0
    assert all(row.shape == (3,) and row.dtype == np.float64 and row.base is None for row in b[3])
    for x, y in zip(saved, b[3]):
        np.testing.assert_array_equal(x, y)


def test_material_provider_legacy_and_snapshot_equivalence():
    mat = material(f=.7, a=.1, s=.2)
    data = mat.get_transport_data()

    class Provider:
        def get_transport_data(self):
            return data

    class LegacyMaterial:
        sigma_f, sigma_a, sigma_s, nu = mat.sigma_f, mat.sigma_a, mat.sigma_s, mat.nu
        name = mat.name

        def validate(self):
            mat.validate()

        def mean_free_path(self):
            return mat.mean_free_path()

    config = dict(n_gen=3, n_source=100, radius=1e9, seed=43,
                  n_burnin=1, max_steps=8, max_samples_per_gen=4)
    reference = physics.estimate_k_full(material=mat, **config)
    assert data.mean_free_path == mat.mean_free_path()
    for source in (Provider(), LegacyMaterial(), data):
        result = physics.estimate_k_full(material=source, **config)
        for key in ('k_history', 'k_mean', 'k_std', 'generation_stats', 'warnings'):
            assert result[key] == reference[key]
        for a, b in zip(result['fiss_frames'], reference['fiss_frames']):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(result['path_frames'], reference['path_frames']):
            for (pa, fa), (pb, fb) in zip(a, b):
                np.testing.assert_array_equal(pa, pb)
                assert fa == fb


def test_material_data_resolved_once_and_run_metadata_owns_values():
    mat = material(f=1., s=0., nu=1.)
    config = dict(n_gen=3, n_source=12, radius=1e9, seed=43, n_burnin=1)
    reference = physics.estimate_k_full(material=mat, **config)
    with patch.object(mat, 'get_transport_data', wraps=mat.get_transport_data) as getter:
        result = physics.estimate_k_full(material=mat, **config,
            callback=lambda *args: setattr(mat, 'nu', 0.))
    getter.assert_called_once_with()
    assert result['k_history'] == reference['k_history'] == [1., 1., 1.]
    assert result['run_info']['material']['nu'] == 1.
    assert result['run_info']['material']['data_source'] == 'constant_parameters'
    assert result['run_info']['seed'] == 43
    assert result['run_info']['backend'] == ('numba' if physics.HAS_NUMBA else 'python')
    assert result['estimator']['active_generation_start'] == 1
    assert result['estimator']['active_generation_count'] == 2
    assert result['estimator']['validated_keff'] is False
    assert result['warnings'] == []


@pytest.mark.parametrize('changes', [dict(sigma_f=-1.), dict(sigma_a=np.nan),
    dict(sigma_s=np.inf), dict(sigma_s=0.), dict(nu=-1.), dict(mean_free_path=0.)])
def test_invalid_transport_data(changes):
    config = dict(sigma_f=0., sigma_a=0., sigma_s=1., nu=0., mean_free_path=1.)
    with pytest.raises(ValueError):
        TransportData(**(config | changes))


def test_snapshot_is_immutable_and_invalid_provider_fails_before_transport():
    from dataclasses import FrozenInstanceError

    data = material().get_transport_data()
    assert resolve_transport_data(data) is data
    with pytest.raises(FrozenInstanceError):
        data.nu = 10.

    class InvalidProvider:
        def get_transport_data(self):
            return np.ones((4, 3))

    with patch.object(physics, 'transport_batch', side_effect=AssertionError('transport entered')):
        with pytest.raises(TypeError, match='must return TransportData'):
            physics.estimate_k_full(material=InvalidProvider())


@pytest.mark.parametrize('config,codes', [
    (dict(n_source=0), ['no_estimate']),
    (dict(n_gen=0), ['no_estimate']),
    (dict(n_gen=1, max_steps=0), ['step_limit_reached', 'burnin_not_applied']),
    (dict(n_gen=1, max_steps=1), ['burnin_not_applied']),
])
def test_run_warning_and_count_contract(config, codes):
    result = physics.estimate_k_full(material=material(a=1., s=0.), seed=2, **config)
    assert result['warnings'] == codes
    assert result['processed_histories'] == sum(s['processed'] for s in result['generation_stats'])
    assert result['processed_histories'] == (result['completed_histories'] +
                                             result['step_limited_histories'])
    assert result['histories_complete'] == ('step_limit_reached' not in codes)
    assert result['estimator']['active_generation_start'] == 0
    assert result['estimator']['active_generation_count'] == len(result['k_history'])
    assert result['estimator']['validated_keff'] is False


def synthetic_dataset(**changes):
    """Deliberately invented fixtures, never evaluated or experimental data."""
    config = dict(material_id='test:synthetic', name='Synthetic fixture',
                  data_kind='synthetic', source='unit-test invented constants',
                  version='1', provenance=('Artificial software test; no scientific provenance.',),
                  quantities=tuple(MaterialQuantity(name, unit, value) for name, unit, value in (
                      ('sigma_f', 'barn', .7), ('sigma_a', 'barn', .1),
                      ('sigma_s', 'barn', .2), ('nu', '1', 2.43), ('mean_free_path', 'cm', 1.))))
    return MaterialDataset(**(config | changes))


def test_pointwise_schema_and_immutable_array_ownership():
    grid = np.array([1., 2., 4.])
    values = np.array([.1, .2, .3])
    dataset = synthetic_dataset(energy_representation='pointwise', energy_grid=grid,
        quantities=(MaterialQuantity('sigma_f', 'barn', values),
                    MaterialQuantity('nu', None, status='missing'),
                    MaterialQuantity('future_channel', 'barn', status='unsupported')))
    grid[:] = values[:] = 0.
    np.testing.assert_array_equal(dataset.energy_grid, [1., 2., 4.])
    np.testing.assert_array_equal(dataset.quantities[0].values, [.1, .2, .3])
    for array in (dataset.energy_grid, dataset.quantities[0].values):
        assert array.dtype == np.float64 and array.flags.c_contiguous
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with patch.object(physics, 'transport_batch', side_effect=AssertionError('transport entered')):
        with pytest.raises(ValueError, match='storage-only'):
            physics.estimate_k_full(material=dataset)


def test_dataset_rejects_inconsistent_arrays_and_energy_descriptions():
    for grid, values, message in (
        (np.array([1., 1.]), np.ones(2), 'strictly increasing'),
        (np.array([2., 1.]), np.ones(2), 'strictly increasing'),
        (np.array([1., np.nan]), np.ones(2), 'finite'),
        (np.array([-1., 2.]), np.ones(2), 'nonnegative'),
        (np.ones((2, 2)), np.ones(2), 'one-dimensional'),
        (np.array([1., 2.]), np.ones(3), 'match energy_grid'),
        (np.array([1., 2.]), 1., 'match energy_grid'),
    ):
        with pytest.raises(ValueError, match=message):
            synthetic_dataset(energy_representation='pointwise', energy_grid=grid,
                              quantities=(MaterialQuantity('sigma_f', 'barn', values),))
    with pytest.raises(TypeError, match='numeric NumPy array'):
        synthetic_dataset(energy_representation='pointwise', energy_grid=np.array(['1', '2']))
    with pytest.raises(ValueError, match='energy_grid=None'):
        synthetic_dataset(energy_grid=np.array([1., 2.]))
    with pytest.raises(ValueError, match='must be scalar'):
        synthetic_dataset(quantities=(MaterialQuantity('sigma_f', 'barn', np.ones(2)),))


def test_quantity_units_missing_values_and_unsupported_channels():
    for config, message in (
        (dict(name='sigma_f', unit='cm', values=1.), 'unit must be barn'),
        (dict(name='sigma_f', unit='mb', values=1.), 'unsupported unit'),
        (dict(name='sigma_f', unit='barn'), 'require unit and values'),
        (dict(name='nu', unit='1', values=0., status='missing'), 'values=None'),
        (dict(name='nu', unit='1', values=np.nan), 'finite'),
        (dict(name='nu', unit='1', values=-1.), 'nonnegative'),
    ):
        with pytest.raises(ValueError, match=message):
            MaterialQuantity(**config)
    for value in (True, '1', 1j, np.array([True]), np.array(['1']),
                  np.ma.array([1.], mask=[True])):
        with pytest.raises(TypeError):
            MaterialQuantity('nu', '1', value)
    dataset = synthetic_dataset(quantities=(MaterialQuantity('sigma_f', None, status='missing'),))
    with pytest.raises(ValueError, match='requires available quantities: sigma_f'):
        resolve_transport_data(dataset)
    quantities = synthetic_dataset().quantities
    with pytest.raises(ValueError, match='does not support available quantities: extra_process'):
        synthetic_dataset(quantities=quantities + (MaterialQuantity('extra_process', 'barn', 1.),)).get_transport_data()
    with pytest.raises(ValueError, match='names must be unique'):
        synthetic_dataset(quantities=quantities + quantities[:1])


def test_material_identity_composition_and_provenance_validation():
    dataset = synthetic_dataset(composition=(('component-A', .25), ('component-B', .75)),
                                composition_basis='atom_fraction', temperature=300.)
    assert dataset.composition_basis == 'atom_fraction' and dataset.temperature_unit == 'K'
    assert dataset.data_kind == 'synthetic' and dataset.version == '1'
    for changes, message in (
        (dict(material_id=''), 'material_id'),
        (dict(material_id='not unique id'), 'whitespace'),
        (dict(data_kind='real'), 'data_kind'),
        (dict(source=None), 'source'),
        (dict(version=''), 'version'),
        (dict(provenance=()), 'provenance'),
        (dict(temperature=-1.), 'temperature'),
        (dict(temperature_unit='C'), 'must be K'),
        (dict(energy_unit='MeV'), 'must be eV'),
        (dict(composition_basis='mass_fraction'), 'missing composition'),
        (dict(composition=(('A', 1.),)), 'explicit basis'),
        (dict(composition=(('A', .3),), composition_basis='mass_fraction'), 'sum to 1'),
        (dict(composition=(('A', .5), ('A', .5)), composition_basis='atom_fraction'), 'unique'),
        (dict(composition=(('A', 1e308), ('B', 1e308)), composition_basis='stoichiometric'), 'finite sum'),
    ):
        with pytest.raises(ValueError, match=message):
            synthetic_dataset(**changes)
    assert synthetic_dataset(composition=(('A', 2.), ('B', 1.)),
                             composition_basis='stoichiometric').composition == (('A', 2.), ('B', 1.))


def test_existing_presets_have_honest_metadata_and_identical_transport_inputs():
    ids = set()
    for key, mat in MATERIALS.items():
        dataset = get_builtin_dataset(key)
        assert dataset.material_id not in ids
        ids.add(dataset.material_id)
        assert dataset.data_kind == 'legacy' and dataset.version is None
        assert dataset.quality.validation_status == 'unverified'
        assert dataset.composition is dataset.temperature is dataset.energy_grid is None
        assert dataset.composition_basis == 'unknown'
        old, new = mat.get_transport_data(), resolve_transport_data(dataset)
        for field in ('sigma_f', 'sigma_a', 'sigma_s', 'nu', 'mean_free_path', 'name'):
            assert getattr(old, field) == getattr(new, field)
    key = next(iter(MATERIALS))
    with patch.object(MATERIALS[key], 'nu', 1.25):
        assert get_builtin_dataset(key).get_transport_data().nu == 1.25
    with pytest.raises(ValueError, match='unknown built-in'):
        get_builtin_dataset('not-a-preset')


def test_structured_one_group_dataset_runs_through_worker_without_numerical_changes():
    dataset = synthetic_dataset()
    config = dict(n_gen=3, n_source=80, radius=1e9, max_steps=8,
                  max_samples_per_gen=4, seed=2026)
    expected = physics.estimate_k_full(material=dataset.get_transport_data(), **config)
    worker = SimulationWorker()
    worker.start(physics.estimate_k_full, material=dataset, **config)
    worker.thread.join(timeout=30)
    assert not worker.thread.is_alive()
    state, result = worker.poll()
    assert state == ExecutionState.COMPLETED
    for key in ('k_history', 'k_mean', 'k_std', 'generation_stats', 'warnings', 'run_info'):
        assert result[key] == expected[key]
    assert 'synthetic:test:synthetic@1' in result['run_info']['material']['data_source']
    for a, b in zip(result['fiss_frames'], expected['fiss_frames']):
        np.testing.assert_array_equal(a, b)
