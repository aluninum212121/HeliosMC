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
from material import Material, TransportData, resolve_transport_data
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
                                                 (2, 100, 'ПОДКРИТИЧНЫЙ'),
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
            assert 'ПОДКРИТИЧНЫЙ' not in text
            assert 'Лимит шагов: 100/100 (100.0%)' in text
        marked = any(artist.get_label() == 'Лимит шагов'
                     for artist in app.ax_k.collections)
        assert marked == (steps == 0)
        app.fig.canvas.draw()
        assert all(t.get_position()[1] > 0 for t in app.ax_stat.texts)
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
    source = Path('benchmarks/results/baseline_physics.txt').read_bytes()
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
