import os
os.environ['MPLBACKEND'] = 'Agg'
import gc
import time
import tracemalloc
from threading import Event, get_ident
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

import physics
from physics import transport_batch, estimate_k_full, SimulationCancelled, HistoryStatus
from material import MATERIALS, Material
from neutron import Neutron
from simulation_worker import SimulationWorker, ExecutionState
from gui import NeutronSimGUI


def particles(n=120):
    return np.zeros((n, 3)), np.tile([0., 0., 1.], (n, 1))


def transport(n=120, **options):
    return transport_batch(*particles(n), 1., 0., 0., 1., 0., 1e9, **options)


@pytest.mark.parametrize('steps', [0, 1, 12])
def test_terminal_limit(steps):
    result = transport(max_steps=steps, seed=4, max_samples=7)
    assert result[2].dtype == np.int32
    assert np.all(result[2] == HistoryStatus.MAX_STEPS_REACHED)
    assert len(result[4]) == 7
    assert all(p.shape == (steps+1, 3) and p.dtype == np.float64 for p in result[4])
    assert all(p.base is None for p in result[4])


@pytest.mark.parametrize('limit', [0, 1, 13, 200])
def test_sampling_independence(limit):
    a = transport(seed=42, max_steps=10, max_samples=limit)
    b = transport(seed=42, max_steps=10, max_samples=0)
    for x, y in zip(a[:3], b[:3]):
        np.testing.assert_array_equal(x, y)
    assert len(a[4]) == min(limit, 120)


def test_seeded_generations():
    a = estimate_k_full(n_gen=5, n_source=300, seed=12)
    b = estimate_k_full(n_gen=5, n_source=300, seed=12, max_samples_per_gen=0)
    c = estimate_k_full(n_gen=5, n_source=300, seed=12)
    d = estimate_k_full(n_gen=5, n_source=300, seed=13)
    assert a.keys() == b.keys() == c.keys() == d.keys()
    assert a['generation_stats'] == b['generation_stats'] == c['generation_stats']
    assert a['k_history'] == b['k_history'] == c['k_history']
    assert a['generation_stats'] != d['generation_stats']
    assert all(not paths for paths in b['path_frames'])
    for x, y in zip(a['fiss_frames'], b['fiss_frames']):
        np.testing.assert_array_equal(x, y)
        assert x.ndim == 2 and x.shape[1] == 3
    for ga, gc_ in zip(a['path_frames'], c['path_frames']):
        for (pa, fa), (pc, fc) in zip(ga, gc_):
            np.testing.assert_array_equal(pa, pc)
            assert fa == fc
    for stats in a['generation_stats']:
        assert sum(stats['outcome_counts'].values()) == stats['processed']


@pytest.mark.parametrize('config', [dict(n_gen=-1), dict(n_source=1.2), dict(radius=0),
    dict(radius=np.nan), dict(radius=np.inf), dict(max_steps=-1), dict(seed=-2),
    dict(max_samples_per_gen=-1), dict(n_burnin=True)])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        estimate_k_full(**config)


@pytest.mark.parametrize('p,d', [(np.zeros((3,)), np.zeros((3,))),
    (np.zeros((2, 3)), np.zeros((3, 3))),
    (np.full((1, 3), np.nan), np.ones((1, 3))),
    (np.zeros((1, 3)), np.zeros((1, 3)))])
def test_invalid_arrays(p, d):
    with pytest.raises(ValueError):
        transport_batch(p, d, 1., 1., 1., 1., 2., 25.)


def test_empty_and_no_mutation():
    a = transport(n=0, seed=3)
    assert a[0].shape == (0, 3) and a[2].shape == (0,) and a[4] == []
    assert estimate_k_full(n_source=0)['k_history'] == []
    assert estimate_k_full(n_gen=0)['generation_stats'] == []
    p, d = particles()
    transport_batch(p, d, 1., 1., 1., 1., 2., 25., seed=2)
    np.testing.assert_array_equal(p, particles()[0])
    np.testing.assert_array_equal(d, particles()[1])


def test_public_validation():
    with pytest.raises(ValueError):
        Material('bad', 0, 0, 0, 0, 1, 1, 'red').mean_free_path()
    with pytest.raises(ValueError):
        physics.sample_distances_batch(np.nan, 3)
    with pytest.raises(ValueError):
        physics.sample_reactions_batch(0., 0., 0., 3)
    with pytest.raises(ValueError):
        physics.scatter_directions(np.zeros((2, 3)), 3)
    with pytest.raises(ValueError):
        Neutron([0, 0, 0], [0, 0, 0])
    n = Neutron([0, 0, 0], [0, 0, 1])
    with pytest.raises(ValueError):
        n.move(np.inf)
    with pytest.raises(ValueError):
        n.scatter(2, 0)


def test_memory_and_release():
    transport(n=2, seed=1, max_steps=30, max_samples=8)  # warm JIT
    gc.collect()
    tracemalloc.start()
    for _ in range(5):
        result = transport(n=4000, seed=1, max_steps=30, max_samples=8)
        assert sum(p.nbytes for p in result[4]) == 8*31*3*8
        del result
    gc.collect()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 12*2**20
    assert current < 2*2**20


def await_worker(worker):
    worker.thread.join(timeout=30)
    assert not worker.thread.is_alive()
    return worker.poll()


def test_worker_states_overlap_cancel_failure():
    worker = SimulationWorker()
    assert worker.state == ExecutionState.IDLE
    gate = Event()
    def job(cancel_event):
        gate.wait(timeout=5)
        return {'ok': True}
    worker.start(job)
    assert worker.state == ExecutionState.RUNNING
    with pytest.raises(RuntimeError, match='already running'):
        worker.start(job)
    worker.cancel()
    gate.set()
    assert await_worker(worker) == (ExecutionState.CANCELLED, None)
    def fail(cancel_event):
        raise ValueError('synthetic error')
    worker.start(fail)
    state, error = await_worker(worker)
    assert state == ExecutionState.FAILED and 'ValueError: synthetic error' in error
    worker.start(job)
    assert await_worker(worker) == (ExecutionState.COMPLETED, {'ok': True})
    assert worker.poll() is None


def test_cancel_checkpoint():
    event = Event()
    event.set()
    with pytest.raises(SimulationCancelled):
        estimate_k_full(cancel_event=event)
    event.clear()
    with pytest.raises(SimulationCancelled):
        estimate_k_full(n_gen=4, seed=1, cancel_event=event, callback=lambda *args: event.set())


def test_gui_completion_failure_cancellation_and_render(tmp_path):
    app = NeutronSimGUI(show=False)
    main_thread = get_ident()
    result = estimate_k_full(n_gen=3, seed=6, max_samples_per_gen=0)
    called = []
    def job(**config):
        called.append(get_ident())
        return result
    try:
        with patch('gui.estimate_k_full', side_effect=job):
            app._on_run(None)
            thread = app.worker.thread
            app._on_run(None)
            assert app.worker.thread is thread
            assert not app.btn_run.active
            thread.join(timeout=30)
            app._poll_worker()
        assert called and called[0] != main_thread
        assert app.execution_state == ExecutionState.COMPLETED and app.btn_run.active
        app.current_gen = 999
        app._render_all()
        app._step_gen(1)
        app.fig.savefig(tmp_path/'gui.png')
        with patch('gui.estimate_k_full', side_effect=ValueError('GUI test failure')):
            app._on_run(None)
            app.worker.thread.join(timeout=30)
            app._poll_worker()
        assert app.execution_state == ExecutionState.FAILED
        assert app.result is None and app.btn_run.active
        assert 'GUI test failure' in app.ax_stat.texts[0].get_text()
        gate = Event()
        def slow(**config):
            gate.wait(timeout=5)
            return result
        with patch('gui.estimate_k_full', side_effect=slow):
            app._on_run(None)
            app.worker.cancel()
            gate.set()
            app.worker.thread.join(timeout=30)
            app._poll_worker()
        assert app.execution_state == ExecutionState.CANCELLED
        assert app.result is None and app.btn_run.active
        app.result = estimate_k_full(n_source=0)
        app._render_all()
        app._step_gen(1)
        app._on_anim(None)
    finally:
        app._on_close(None)
        plt.close(app.fig)


def test_entry_point():
    import main
    with patch('main.NeutronSimGUI', return_value='GUI'):
        assert main.main() == 'GUI'


def test_optional_numba_fallback():
    import subprocess
    import sys
    code = '''
import sys
sys.modules['numba'] = None
import numpy as np
import physics
assert not physics.HAS_NUMBA
x = physics.estimate_k_full(n_gen=3, n_source=40, seed=9)
y = physics.estimate_k_full(n_gen=3, n_source=40, seed=9, max_samples_per_gen=0)
assert x['generation_stats'] == y['generation_stats']
assert x['k_history'] == y['k_history']
assert all(not p for p in y['path_frames'])
'''
    subprocess.run([sys.executable, '-c', code], check=True, timeout=60)


def test_close_requests_worker_shutdown():
    app = NeutronSimGUI(show=False)
    entered = Event()
    def job(cancel_event, **config):
        entered.set()
        assert cancel_event.wait(timeout=5)
        raise SimulationCancelled('closed')
    try:
        with patch('gui.estimate_k_full', side_effect=job):
            app._on_run(None)
            assert entered.wait(timeout=5)
            app._on_close(None)
            assert await_worker(app.worker) == (ExecutionState.CANCELLED, None)
        assert app._closed and app.worker.thread is None
    finally:
        plt.close(app.fig)


def test_worker_start_failure_and_gui_render_failure():
    worker = SimulationWorker()
    with patch('simulation_worker.Thread.start', side_effect=RuntimeError('cannot start')):
        with pytest.raises(RuntimeError, match='cannot start'):
            worker.start(lambda **kw: None)
    assert worker.state == ExecutionState.FAILED
    app = NeutronSimGUI(show=False)
    try:
        result = estimate_k_full(n_gen=1, seed=1)
        with patch('gui.estimate_k_full', return_value=result):
            app._on_run(None)
            app.worker.thread.join(timeout=30)
            with patch.object(app, '_render_all', side_effect=ValueError('render error')):
                app._poll_worker()
        assert app.execution_state == ExecutionState.FAILED
        assert app.btn_run.active and app.result is None
        assert 'render error' in app.ax_stat.texts[0].get_text()
    finally:
        app._on_close(None)
        plt.close(app.fig)


def test_sampler_shapes_and_sizes():
    for function, args in [(physics.scatter_directions, (np.zeros(3), 1)),
                           (physics.sample_distances_batch, (1., 1.5)),
                           (physics.sample_reactions_batch, (1., 1., 1., -1))]:
        with pytest.raises(ValueError):
            function(*args)
