"""Independent mathematical checks for nonreactive educational scenarios."""
from dataclasses import replace
from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
from scipy import integrate, stats

import physics
from material import (MaterialDataset, MaterialQuantity, resolve_medium_rates,
                      interpolate_synthetic_quantity)
from simulation_worker import SimulationWorker, ExecutionState


@physics.njit
def seed_legacy_sampler(seed):
    np.random.seed(seed)


def medium(a=0., s=0.):
    return MaterialDataset('test:abstract', 'Invented abstract medium', 'synthetic',
        'software fixture, not nuclear data', '1', ('Artificial dimensioned rates.',),
        (MaterialQuantity('absorption_rate', '1/cm', a),
         MaterialQuantity('scattering_rate', '1/cm', s)))


@pytest.mark.parametrize('rates,widths', [([.8], [2.]), ([.2, 0., 1.1], [.5, 1., .75])])
def test_absorption_matches_analytic_attenuation_and_quadrature(rates, widths):
    edges = np.r_[0., np.cumsum(widths)]
    expected = np.exp(-np.dot(rates, widths))
    result = physics.simulate_slab([medium(a) for a in rates], edges,
        n_histories=30000, seed=931, max_collisions=1, max_samples=5)
    assert result['unfinished_histories'] == 0
    success = result['outcome_counts']['escaped_right']
    # Fixed seed and a predeclared conservative two-sided alpha, not exact RNG values.
    assert stats.binomtest(success, 30000, expected).pvalue > 1e-6
    assert result['outcome_counts']['escaped_left'] == 0
    absorbed = 0.
    upstream_depth = 0.
    for rate, width in zip(rates, widths):
        absorbed += integrate.quad(lambda x: rate * np.exp(-upstream_depth - rate*x), 0., width)[0]
        upstream_depth += rate * width
    assert absorbed + expected == pytest.approx(1., abs=1e-12)
    assert sum(result['outcome_counts'].values()) == 30000
    for path in result['sampled_paths_cm']:
        assert np.all((path >= 0) & (path <= edges[-1]))


def test_vacuum_interfaces_source_boundaries_empty_and_budget():
    r = physics.simulate_slab([medium(), medium()], [0., 1., 3.],
        n_histories=20, max_collisions=0, max_samples=3, seed=4)
    assert r['outcome_counts']['escaped_right'] == 20
    assert all(path.tolist() == [0., 3.] for path in r['sampled_paths_cm'])
    left = physics.simulate_slab([medium()], [0., 1.], n_histories=8, source_mu=-1., seed=4)
    assert left['outcome_counts']['escaped_left'] == 8
    empty = physics.simulate_slab([medium()], [0., 1.], n_histories=0, seed=1)
    assert empty['warnings'] == ['no_histories']
    assert empty['estimates']['absorbed']['unavailable_reason'] == 'empty_sample'
    with pytest.raises(MemoryError):
        physics.simulate_slab([medium()], [0., 1.], max_samples=10, max_memory_bytes=1)


def test_censoring_never_produces_completed_only_estimates():
    r = physics.simulate_slab([medium(2., 4.)], [0., 1.],
        n_histories=1000, max_collisions=0, seed=14)
    assert 0 < r['unfinished_histories'] < 1000
    assert r['completed_histories'] + r['unfinished_histories'] == 1000
    for estimate in r['estimates'].values():
        assert estimate['estimate'] is estimate['confidence_interval'] is estimate['standard_error'] is None
        assert estimate['unavailable_reason'] == 'censored_histories'
        lo, hi = estimate['completion_bounds']
        assert 0 <= lo <= hi <= 1


def test_seed_chunk_sample_backend_independence_and_global_rng_ownership():
    config = dict(layers=[medium(.2, .7), medium(.5, .3)], edges_cm=[0., 1., 3.],
                  n_histories=1500, max_collisions=100, seed=56)
    np.random.seed(731)
    state = np.random.get_state()
    a = physics.simulate_slab(**config, batch_size=41, max_samples=7, backend='python')
    b = physics.simulate_slab(**config, batch_size=512, max_samples=0)
    c = physics.simulate_slab(**config, batch_size=99, max_samples=7)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    assert a['outcome_counts'] == b['outcome_counts'] == c['outcome_counts']
    assert a['estimates'] == b['estimates'] == c['estimates']
    for p, q in zip(a['sampled_paths_cm'], c['sampled_paths_cm']):
        np.testing.assert_allclose(p, q, rtol=1e-13, atol=1e-13)
    unseeded = physics.simulate_slab([medium(.5)], [0., 1.], n_histories=200)
    replay = physics.simulate_slab([medium(.5)], [0., 1.], n_histories=200,
                                  seed=unseeded['run_info']['effective_seed'])
    assert unseeded['outcome_counts'] == replay['outcome_counts']


def test_scattering_symmetry_and_no_absorption():
    config = dict(layers=[medium(s=2.)], edges_cm=[0., 2.], n_histories=12000,
                  max_collisions=500, source_x=1.)
    plus = physics.simulate_slab(**config, source_mu=1., seed=9)
    minus = physics.simulate_slab(**config, source_mu=-1., seed=10)
    assert plus['histories_complete'] and minus['histories_complete']
    assert plus['outcome_counts']['absorbed'] == minus['outcome_counts']['absorbed'] == 0
    # Reflection invariance: mirrored initial directions exchange exit boundaries.
    p = plus['estimates']['escaped_right']['estimate']
    q = minus['estimates']['escaped_left']['estimate']
    se = np.sqrt(p*(1-p)/12000 + q*(1-q)/12000)
    assert abs(p-q) < 6*se


def test_identical_layer_partition_and_oblique_attenuation():
    config = dict(n_histories=2000, seed=124, max_samples=8, max_collisions=100)
    one = physics.simulate_slab([medium(.3, .5)], [0., 2.], **config)
    split = physics.simulate_slab([medium(.3, .5)]*2, [0., .75, 2.], **config)
    assert one['outcome_counts'] == split['outcome_counts']
    for a, b in zip(one['sampled_paths_cm'], split['sampled_paths_cm']):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    tilted = physics.simulate_slab([medium(.5)], [0., 1.], source_mu=.5,
                                  n_histories=10000, max_collisions=1, seed=19)
    assert stats.binomtest(tilted['outcome_counts']['escaped_right'], 10000, np.exp(-1.)).pvalue > 1e-6


def test_numerical_limits_and_statistical_invalid_domains_are_explicit():
    r = physics.simulate_slab([medium(1e308)], [0., 2.], source_x=1.,
                             n_histories=5, max_collisions=2, seed=2)
    assert r['outcome_counts']['numeric_limit'] == 5
    assert r['estimates']['absorbed']['confidence_interval'] is None
    assert r['warnings'] == ['numerical_resolution_limit']
    for args in ((2, 1), (-1, 2), (0, 0, 1.), (0, 2**64)):
        with pytest.raises(ValueError):
            physics.summarize_binomial(*args)
    with pytest.raises(ValueError, match='underflows'):
        physics.summarize_samples([1e-300, 2e-300])
    stable = physics.summarize_samples(np.array([1e12, 1e12+1, 1e12+2]))
    assert stable['mean'] == 1e12+1 and stable['sample_variance'] == 1.


def test_empirical_monte_carlo_error_scales_with_inverse_square_root_n():
    rng = np.random.default_rng(991)
    p = .3
    mse = []
    for n in (100, 400):
        estimates = rng.binomial(n, p, size=2000)/n
        mse.append(np.mean((estimates-p)**2))
        assert mse[-1] == pytest.approx(p*(1-p)/n, rel=.12)
    assert np.sqrt(mse[0]/mse[1]) == pytest.approx(2., rel=.12)


def test_free_path_distribution_without_epsilon_and_isotropic_cosines():
    # In one absorbing layer collision coordinates sample Exp(rate), right-censored at width.
    r = physics.simulate_slab([medium(1.)], [0., 40.], n_histories=10000,
        max_samples=10000, max_collisions=1, seed=111, max_memory_bytes=8*2**20)
    assert r['outcome_counts']['absorbed'] == 10000
    distances = np.array([path[-1] for path in r['sampled_paths_cm']])
    assert stats.kstest(distances, 'expon').pvalue > 1e-6
    assert distances.mean() == pytest.approx(1., abs=.04)
    # Existing standalone isotropic helper: first and second angular moments.
    np.random.seed(123)
    seed_legacy_sampler(123)
    directions = physics.scatter_directions(np.tile([0., 0., 1.], (10000, 1)), 10000)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-13)
    assert abs(directions[:, 2].mean()) < .04
    assert abs(np.mean(directions[:, 2]**2) - 1/3) < .03


def test_binomial_exact_endpoints_coverage_and_sample_size_scaling():
    for k in (0, 10):
        result = physics.summarize_binomial(k, 10)
        lo, hi = result['confidence_interval']
        if k == 0:
            assert lo == 0 and hi == pytest.approx(1 - .025**.1)
        else:
            assert hi == 1 and lo == pytest.approx(.025**.1)
    # Enumerated frequentist coverage, not comparison to the same interval routine.
    intervals = [physics.summarize_binomial(k, 40)['confidence_interval'] for k in range(41)]
    for p in (.01, .1, .5, .9, .99):
        coverage = sum(stats.binom.pmf(k, 40, p) for k, (lo, hi) in enumerate(intervals) if lo <= p <= hi)
        assert coverage >= .95 - 1e-12
    small = physics.summarize_binomial(50, 100)
    large = physics.summarize_binomial(200, 400)
    assert large['standard_error'] == pytest.approx(small['standard_error']/2)
    for n, k in ((20, 3), (100, 50), (2000, 1700)):
        lo, hi = physics.summarize_binomial(k, n)['confidence_interval']
        assert stats.binom.sf(k-1, n, lo) == pytest.approx(.025, abs=1e-11)
        assert stats.binom.cdf(k, n, hi) == pytest.approx(.025, abs=1e-11)
        reference = stats.binomtest(k, n).proportion_ci(.95, method='exact')
        np.testing.assert_allclose([lo, hi], [reference.low, reference.high], atol=1e-11, rtol=1e-11)


def test_student_mean_interval_and_independent_normal_coverage():
    x = np.array([1., 2., 4., 5.])
    r = physics.summarize_samples(x, assumption='iid_normal')
    assert r['mean'] == 3.
    assert r['sample_variance'] == pytest.approx(10/3)
    assert r['standard_error'] == pytest.approx(np.sqrt(10/12))
    assert r['confidence_interval'][1] == pytest.approx(3 + 3.182446305284263*np.sqrt(10/12))
    rng = np.random.default_rng(173)
    covered = 0
    for row in rng.normal(2., 1., size=(400, 20)):
        lo, hi = physics.summarize_samples(row, assumption='iid_normal')['confidence_interval']
        covered += lo <= 2 <= hi
    assert stats.binomtest(covered, 400, .95).pvalue > 1e-6
    assert physics.summarize_samples(x)['standard_error'] is None
    for values in ([], [1.], [1., 1.]):
        assert physics.summarize_samples(values, assumption='iid_normal')['confidence_interval'] is None
    for values in ([np.nan], [np.inf], [True], ['1'], [[1., 2.]], [1e308, -1e308]):
        with pytest.raises(ValueError):
            physics.summarize_samples(values)


def test_synthetic_pchip_linear_exact_monotone_and_bounded_error():
    grid = np.linspace(0., 2., 17)
    dataset = MaterialDataset('test:table', 'Synthetic polynomial', 'synthetic',
        'analytic example', '1', ('Invented function f(E)=E**2.',),
        (MaterialQuantity('synthetic_response', '1', grid**2),),
        energy_representation='pointwise', energy_grid=grid)
    query = np.linspace(0., 2., 301)
    result = interpolate_synthetic_quantity(dataset, 'synthetic_response', query)
    assert np.all(np.diff(result) >= 0)
    assert np.max(abs(result-query**2)) < .003
    linear = replace(dataset, quantities=(MaterialQuantity('synthetic_response', '1', 2*grid+1),))
    np.testing.assert_allclose(interpolate_synthetic_quantity(linear, 'synthetic_response', query), 2*query+1, atol=1e-13)
    with pytest.raises(ValueError, match='extrapolation'):
        interpolate_synthetic_quantity(dataset, 'synthetic_response', np.array([3.]))
    with pytest.raises(ValueError, match='synthetic'):
        interpolate_synthetic_quantity(replace(dataset, data_kind='evaluated'), 'synthetic_response', query)


@pytest.mark.parametrize('changes', [dict(n_histories=-1), dict(n_histories=True), dict(batch_size=0),
    dict(source_mu=0.), dict(source_x=1.), dict(confidence=1.), dict(edges_cm=[0., 0.]),
    dict(edges_cm=[0., np.inf]), dict(max_memory_bytes=True)])
def test_invalid_slab_inputs(changes):
    with pytest.raises((ValueError, MemoryError)):
        physics.simulate_slab(**(dict(layers=[medium()], edges_cm=[0., 1.], n_histories=1) | changes))


def test_medium_units_and_legacy_model_separation():
    assert resolve_medium_rates(medium(.2, .3)) == (.2, .3)
    with pytest.raises(ValueError, match='unit must be 1/cm'):
        MaterialQuantity('absorption_rate', 'barn', 1.)
    with pytest.raises(ValueError, match='synthetic or educational'):
        resolve_medium_rates(replace(medium(), data_kind='experimental'))
    with pytest.raises(ValueError, match='requires available quantities'):
        medium().get_transport_data()


def test_worker_execution_cancellation_and_failure():
    event = Event()
    event.set()
    with pytest.raises(physics.SimulationCancelled):
        physics.simulate_slab([medium()], [0., 1.], cancel_event=event)
    worker = SimulationWorker()
    worker.start(physics.simulate_slab, layers=[medium(.5)], edges_cm=[0., 1.],
                 n_histories=100, seed=14)
    worker.thread.join(timeout=30)
    state, result = worker.poll()
    assert state == ExecutionState.COMPLETED and result['processed_histories'] == 100
    worker.start(physics.simulate_slab, layers=[medium()], edges_cm=[0., 0.])
    worker.thread.join(timeout=30)
    state, error = worker.poll()
    assert state == ExecutionState.FAILED and 'edges_cm' in error
    entered, release = Event(), Event()
    original = getattr(physics._slab_batch, 'py_func', physics._slab_batch)
    def gated(*args):
        entered.set()
        assert release.wait(timeout=5)
        return original(*args)
    try:
        with patch.object(physics, '_slab_batch', gated):
            worker.start(physics.simulate_slab, layers=[medium(.5)], edges_cm=[0., 1.],
                         n_histories=100, batch_size=10, backend='python', seed=9)
            assert entered.wait(timeout=5)
            worker.cancel()
            release.set()
            worker.thread.join(timeout=30)
            assert worker.poll() == (ExecutionState.CANCELLED, None)
    finally:
        release.set()
        worker.cancel()
        if worker.thread:
            worker.thread.join(timeout=30)
