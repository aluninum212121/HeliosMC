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


def test_research_export_roundtrip_replay_and_feature_separation(tmp_path):
    config = physics.create_slab_configuration("roundtrip", [medium(.5)], [0., 2.], 4000,
        seed=123, verification="pure_absorption_v1", max_samples=2)
    experiment = physics.run_slab_experiment(config)
    destination = tmp_path / "experiment"
    digest = physics.export_slab_experiment(experiment, destination)
    loaded = physics.load_slab_experiment(destination, expected_sha256=digest)
    assert loaded == experiment
    replay = physics.run_slab_experiment(loaded["configuration"])
    assert replay["result"]["outcome_counts"] == loaded["result"]["outcome_counts"]
    assert replay["result"]["estimates"] == loaded["result"]["estimates"]
    assert loaded["verification"]["status"] == "passed"
    assert loaded["verification"]["analytic_result"] == pytest.approx(np.exp(-1.))
    assert loaded["verification"]["alpha"] == config["verification_protocol"]["alpha"] == 1e-6
    assert set(loaded["ml_record"]["features"]) == {"rates_per_cm", "edges_cm", "source_x_cm", "source_mu", "n_histories", "max_collisions"}
    assert loaded["ml_record"]["metadata"]["eligible_for_training"]
    assert "sampled_paths_cm" not in loaded["result"]
    with np.load(destination / "arrays.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"rates_per_cm", "edges_cm", "outcome_counts"}
    with pytest.raises(FileExistsError):
        physics.export_slab_experiment(experiment, destination)


def test_research_python_numba_and_seed_none_replay():
    config = physics.create_slab_configuration("unseeded", [medium(.3, .5)], [0., 1.], 500,
        backend="python", seed=None)
    assert config["requested_seed"] is None and isinstance(config["parameters"]["seed"], int)
    one = physics.run_slab_experiment(config)
    two = physics.run_slab_experiment(config)
    assert one["result"]["outcome_counts"] == two["result"]["outcome_counts"]
    assert one["verification"] is None
    if physics.HAS_NUMBA:
        compiled = physics.create_slab_configuration("compiled", [medium(.3, .5)], [0., 1.], 500,
            backend="numba", seed=config["parameters"]["seed"])
        assert physics.run_slab_experiment(compiled)["result"]["outcome_counts"] == one["result"]["outcome_counts"]


@pytest.mark.parametrize("a,n,limit,warning", [(2., 500, 0, "collision_limit_reached"), (0., 0, 1, "no_histories")])
def test_export_censored_empty_runs_never_become_training_targets(tmp_path, a, n, limit, warning):
    config = physics.create_slab_configuration("limits", [medium(a)], [0., 1.], n,
        seed=14, max_collisions=limit, verification="pure_absorption_v1")
    experiment = physics.run_slab_experiment(config)
    assert experiment["verification"]["status"] == "inconclusive"
    assert experiment["verification"]["absolute_error"] is None
    assert warning in experiment["result"]["warnings"]
    assert all(v is None for v in experiment["ml_record"]["targets"].values())
    with pytest.raises(ValueError, match="training requires"):
        physics.split_slab_groups([experiment])
    digest = physics.export_slab_experiment(experiment, tmp_path / "run")
    assert physics.load_slab_experiment(tmp_path / "run", expected_sha256=digest) == experiment


def test_export_numeric_limits_and_backward_boundary_verification(tmp_path):
    config = physics.create_slab_configuration("numeric", [medium(1e308)], [0., 2.], 5,
        source_x=1., seed=2, verification="pure_absorption_v1")
    experiment = physics.run_slab_experiment(config)
    assert experiment["result"]["warnings"] == ["numerical_resolution_limit"]
    assert experiment["verification"]["status"] == "inconclusive"
    physics.export_slab_experiment(experiment, tmp_path / "numeric")
    assert physics.load_slab_experiment(tmp_path / "numeric") == experiment
    for x in (0., .5, 1.):
        c = physics.create_slab_configuration("left", [medium(.2), medium(.4)], [0., 1., 2.], 2000,
            source_x=x, source_mu=-.5, seed=7, verification="pure_absorption_v1")
        r = physics.run_slab_experiment(c)
        assert r["verification"]["analytic_result"] == pytest.approx(np.exp(-.2*x/.5))
        assert r["verification"]["status"] == "passed"


def test_research_groups_include_repeats_despite_seed_population_and_labels():
    experiments = []
    for seed, n, label in ((11, 300, "a"), (22, 500, "b"), (33, 400, "c")):
        config = physics.create_slab_configuration(label, [replace(medium(.5), material_id=label)], [0., 1.], n,
            seed=seed, verification="pure_absorption_v1")
        experiments.append(physics.run_slab_experiment(config))
    assert len({r["configuration"]["configuration_id"] for r in experiments}) == 3
    assert len({r["configuration"]["group_id"] for r in experiments}) == 1
    train, test = physics.split_slab_groups(experiments)
    assert len(train) + len(test) == 3 and (not train or not test)
    assert not ({r["configuration"]["group_id"] for r in train} & {r["configuration"]["group_id"] for r in test})


@pytest.mark.parametrize("member", ["experiment.json", "results.csv", "arrays.npz"])
def test_corrupt_export_members_are_rejected(tmp_path, member):
    c = physics.create_slab_configuration("corrupt", [medium()], [0., 1.], 10, seed=1)
    r = physics.run_slab_experiment(c)
    physics.export_slab_experiment(r, tmp_path / "run")
    path = tmp_path / "run" / member
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError):
        physics.load_slab_experiment(tmp_path / "run")


@pytest.mark.parametrize("change", ["count", "ci", "warning", "features", "verification", "source", "version", "group"])
def test_export_semantic_tampering_is_rejected(tmp_path, change):
    import copy
    from material import _content_hash
    c = physics.create_slab_configuration("tampering", [medium(.5)], [0., 1.], 100, seed=1,
        verification="pure_absorption_v1")
    r = copy.deepcopy(physics.run_slab_experiment(c))
    if change == "count": r["result"]["outcome_counts"]["absorbed"] += 1
    if change == "ci": r["result"]["estimates"]["absorbed"]["confidence_interval"][0] = 0.
    if change == "warning": r["result"]["warnings"] = ["fake"]
    if change == "features": r["ml_record"]["features"]["target"] = .5
    if change == "verification": r["verification"]["analytic_result"] = .5
    if change == "source": r["configuration"]["layers"][0]["data_kind"] = "reference"
    if change == "version": r["configuration"]["algorithm_version"] = "unknown"
    if change == "group": r["configuration"]["group_id"] = "0"*64
    config = r["configuration"]
    config["configuration_id"] = _content_hash({k:v for k,v in config.items() if k != "configuration_id"})
    with pytest.raises(ValueError):
        physics.export_slab_experiment(r, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_npz_object_and_false_csv_are_rejected_even_with_new_checksums(tmp_path):
    import hashlib
    import json
    from material import _content_hash
    c = physics.create_slab_configuration("object", [medium()], [0., 1.], 10, seed=1)
    r = physics.run_slab_experiment(c)
    for name in ("arrays.npz", "results.csv"):
        directory = tmp_path / name.replace(".", "-")
        physics.export_slab_experiment(r, directory)
        if name.endswith("npz"):
            np.savez(directory / name, edges_cm=np.array([object()], dtype=object), rates_per_cm=np.array([[0., 0.]]), outcome_counts=np.array([0,0,10,0,0]))
        else:
            (directory / name).write_bytes(b"false results\n")
        path = directory / "experiment.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["payload"]["sidecar_sha256"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        raw["content_sha256"] = _content_hash({k:v for k,v in raw.items() if k != "content_sha256"})
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="shape/dtype|CSV"):
            physics.load_slab_experiment(directory)


def test_predeclared_slab_ensemble_convergence_and_coefficient_effect():
    # Criteria fixed before execution: independent replicate RMSE scaling within
    # 35% of sqrt(2048/128)=4; binomial checks alpha=1e-6. No monotonic demand
    # on one noisy run and no fitted tolerance from observed outcomes.
    p = np.exp(-.7)
    errors = []
    for n in (128, 2048):
        values = [physics.simulate_slab([medium(.7)], [0., 1.], n, seed=1000+j,
                   max_collisions=1)["estimates"]["escaped_right"]["estimate"] for j in range(48)]
        errors.append(np.sqrt(np.mean((np.asarray(values)-p)**2)))
    assert errors[0]/errors[1] == pytest.approx(4., rel=.35)
    for rate in (0., .2, 1.):
        r = physics.simulate_slab([medium(rate)], [0., 1.], 4000, seed=77, max_collisions=1)
        assert stats.binomtest(r["outcome_counts"]["escaped_right"], 4000, np.exp(-rate)).pvalue > 1e-6


def test_research_rejects_legacy_scientific_and_nonanalytic_configurations():
    from material import get_builtin_dataset, get_reference_dataset, MATERIALS
    for dataset in [get_builtin_dataset(k) for k in MATERIALS] + [get_reference_dataset("Вода (H₂O)")]:
        with pytest.raises(ValueError):
            physics.create_slab_configuration("forbidden", [dataset], [0., 1.])
    with pytest.raises(ValueError, match="zero scattering"):
        physics.create_slab_configuration("wrong oracle", [medium(s=1.)], [0., 1.], verification="pure_absorption_v1")
    with pytest.raises(ValueError, match="unsupported"):
        physics.create_slab_configuration("wrong option", [medium()], [0., 1.], target=3.)


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


def reference_water():
    from material import get_reference_transport_dataset, REFERENCE_APPROXIMATION
    return get_reference_transport_dataset("Вода (H₂O)", approximation=REFERENCE_APPROXIMATION)


def test_reference_rates_reach_unchanged_kernel_and_preserve_default_outputs():
    from material import resolve_reference_rates
    water = reference_water()
    a, s = resolve_reference_rates(water)
    options = dict(n_histories=2000, seed=201, max_collisions=200, max_samples=4, backend="python")
    reference = physics.simulate_slab([water], [0., .2], material_mode="reference", **options)
    explicit = physics.simulate_slab([medium(a,s)], [0., .2], **options)
    legacy = physics.simulate_slab([medium(a,s)], [0., .2], material_mode="legacy", **options)
    for key in ("outcome_counts", "estimates", "sampled_path_fates"):
        assert reference[key] == explicit[key] == legacy[key]
    for p, q in zip(reference["sampled_paths_cm"], explicit["sampled_paths_cm"]):
        np.testing.assert_array_equal(p, q)
    assert explicit["run_info"] == legacy["run_info"]
    assert reference["run_info"]["material_mode"] == "reference"
    assert reference["run_info"]["layers"][0]["quantities"][1]["values"] == water.state.density
    assert "reference_free_gas_fixed_energy_approximation" in reference["warnings"]
    assert "no_bound_atom_thermal_scattering" in reference["limitations"]
    if physics.HAS_NUMBA:
        compiled = physics.simulate_slab([water], [0., .2], material_mode="reference", **(options | dict(backend="numba")))
        assert compiled["outcome_counts"] == reference["outcome_counts"]
        assert compiled["estimates"] == reference["estimates"]


def test_reference_json_csv_npz_replay_metadata_and_grouping(tmp_path):
    water = reference_water()
    experiments = []
    for seed in (701, 702):
        config = physics.create_slab_configuration("reference-demo", [water], [0., .2], 1000,
            seed=seed, material_mode="reference", backend="python")
        experiment = physics.run_slab_experiment(config)
        digest = physics.export_slab_experiment(experiment, tmp_path / str(seed))
        loaded = physics.load_slab_experiment(tmp_path / str(seed), expected_sha256=digest)
        assert loaded == experiment
        replay = physics.run_slab_experiment(loaded["configuration"])
        assert replay["result"]["outcome_counts"] == experiment["result"]["outcome_counts"]
        assert replay["result"]["estimates"] == experiment["result"]["estimates"]
        assert loaded["configuration"]["parameters"]["seed"] == seed
        metadata = loaded["ml_record"]["metadata"]
        assert metadata["material_mode"] == "reference"
        assert metadata["parameter_verification"] == ["source_checked"]
        assert metadata["scientific_validation"] == "not_established"
        assert not metadata["eligible_for_training"]
        assert set(loaded["ml_record"]["features"]) == {"rates_per_cm", "edges_cm", "source_x_cm", "source_mu", "n_histories", "max_collisions"}
        with np.load(tmp_path / str(seed) / "arrays.npz", allow_pickle=False) as arrays:
            assert arrays["rates_per_cm"].tolist() == loaded["ml_record"]["features"]["rates_per_cm"]
        experiments.append(loaded)
    assert experiments[0]["configuration"]["group_id"] == experiments[1]["configuration"]["group_id"]
    with pytest.raises(ValueError, match="training requires"):
        physics.split_slab_groups(experiments)


def test_reference_graphite_density_is_input_not_automatic_validation(tmp_path):
    from material import get_reference_transport_dataset, DataOrigin, REFERENCE_APPROXIMATION
    origin = DataOrigin(original_reference="Invented sample for testing, not an experimental measurement",
        processing_description="Caller supplied 1.6 g/cm^3 at 300 K")
    graphite = get_reference_transport_dataset("Графит", approximation=REFERENCE_APPROXIMATION,
        density=1.6, density_origin=origin)
    c = physics.create_slab_configuration("graphite", [graphite], [0., .1], 100, seed=3, material_mode="reference")
    r = physics.run_slab_experiment(c)
    physics.export_slab_experiment(r, tmp_path / "graphite")
    assert physics.load_slab_experiment(tmp_path / "graphite") == r
    assert r["ml_record"]["metadata"]["parameter_verification"] == ["unverified"]
    assert not r["ml_record"]["metadata"]["eligible_for_training"]


def test_old_v1_experiment_archive_remains_readable(tmp_path):
    from material import _content_hash
    c = physics.create_slab_configuration("old-contract", [medium(.5)], [0., 1.], 200, seed=4, verification="pure_absorption_v1")
    # Reconstruct the exact old schema, including absence of modern metadata.
    c["schema_version"] = "heliosmc.slab-configuration.v1"
    del c["material_mode"], c["material_model_version"], c["reference_profile"]
    c["configuration_id"] = _content_hash({k:v for k,v in c.items() if k != "configuration_id"})
    r = physics.run_slab_experiment(c)
    assert "material_mode" not in r["ml_record"]["metadata"]
    physics.export_slab_experiment(r, tmp_path / "v1")
    assert physics.load_slab_experiment(tmp_path / "v1") == r


@pytest.mark.parametrize("change", ["mode", "energy", "temperature", "model", "warning", "ml_status"])
def test_reference_experiment_rejects_semantic_tampering(tmp_path, change):
    from material import _content_hash
    c = physics.create_slab_configuration("forged-reference", [reference_water()], [0., .1], 30, seed=4, material_mode="reference")
    r = physics.run_slab_experiment(c)
    c = r["configuration"]
    if change == "mode": c["material_mode"] = "legacy"
    if change == "model": c["material_model_version"] = "full_thermal_physics"
    if change == "energy": c["layers"][0]["quantities"][-1]["averaging"]["energy_bounds_ev"] = [.03,.03]
    if change == "temperature":
        c["layers"][0]["temperature"] = 298.15
        c["layers"][0]["state"]["temperature_range"] = [298.15,300.]
    if change == "warning": r["result"]["warnings"] = []
    if change == "ml_status": r["ml_record"]["metadata"]["eligible_for_training"] = True
    c["configuration_id"] = _content_hash({k:v for k,v in c.items() if k != "configuration_id"})
    with pytest.raises(ValueError):
        physics.export_slab_experiment(r, tmp_path / "bad")


def test_reference_no_mode_mixing_or_missing_data_fallback():
    from material import get_reference_dataset, get_builtin_dataset, MATERIALS
    for dataset in [get_reference_dataset("Вода (H₂O)"), medium()] + [get_builtin_dataset(k) for k in MATERIALS]:
        with pytest.raises(ValueError):
            physics.create_slab_configuration("no-fallback", [dataset], [0., 1.], 10, material_mode="reference")
    with pytest.raises(ValueError):
        physics.simulate_slab([reference_water()], [0., 1.], 10)
    with pytest.raises(ValueError):
        physics.simulate_slab([reference_water(),medium()], [0., 1., 2.], 10, material_mode="reference")
    with pytest.raises(ValueError, match="zero scattering"):
        physics.create_slab_configuration("wrong-oracle", [reference_water()], [0., 1.], material_mode="reference", verification="pure_absorption_v1")


def test_reference_first_flight_analytic_ensemble_predeclared():
    # Fixed before running: each two-sided binomial p >= 1e-6 for five independent
    # seeds, plus aggregate p >= 1e-6. max_collisions=0 counts only first-flight
    # escape; collision histories remain censored, not fabricated full solutions.
    from scipy.constants import Avogadro
    water = reference_water()
    sigma_a = 2*(.99985*.3326+.00015*.00055)+.99762*.0001899+.00038*.24447+.002*.0001408
    sigma_s = 2*(.99985*30.27+.00015*4.246)+.99762*3.986+.00038*3.615+.002*3.532
    rate = water.state.density*Avogadro/18.0153*(sigma_a+sigma_s)*1e-24
    expected = float(np.exp(-rate*.3))
    counts = []
    for seed in (101, 202, 303, 404, 505):
        r = physics.simulate_slab([water], [0., .3], 5000, seed=seed, max_collisions=0, material_mode="reference")
        k = r["outcome_counts"]["escaped_right"]
        pvalue = stats.binomtest(k, 5000, expected).pvalue
        counts.append(k)
        print(f"reference-first-flight seed={seed}, right={k}/5000, expected={expected:.12g}, p={pvalue:.12g}")
        assert pvalue >= 1e-6
        assert r["unfinished_histories"] == 5000-k
        assert r["estimates"]["escaped_right"]["estimate"] is None
    assert stats.binomtest(sum(counts), 25000, expected).pvalue >= 1e-6


def graphite_profile(profile_id):
    from material import get_reference_profiles, get_reference_transport_dataset, DataOrigin
    profile = get_reference_profiles()[profile_id]
    return get_reference_transport_dataset("Графит",profile_id=profile_id,approximation=profile["approximation"],
        density=1.6,density_origin=DataOrigin(original_reference="Invented sample for profile software tests",
            processing_description="Caller input 1.6 g/cm^3 at 300 K; no specimen measurement claimed"))


@pytest.mark.parametrize("profile_id", ["jendl5_point_0p0253ev", "jendl40_point_0p0253ev", "jendl40_maxwell_flux_0p0253ev"])
def test_profile_selection_reproducibility_export_and_kernel_equivalence(tmp_path, profile_id):
    from material import resolve_reference_rates, _json_data
    d = graphite_profile(profile_id)
    c = physics.create_slab_configuration("profile-roundtrip",[d],[0.,2.],700,seed=301,
        material_mode="reference",backend="python",max_collisions=100)
    assert c["schema_version"] == "heliosmc.slab-configuration.v3"
    assert c["reference_profile"]["profile_id"] == profile_id
    r = physics.run_slab_experiment(c)
    plain = physics.simulate_slab([medium(*resolve_reference_rates(d))],[0.,2.],700,seed=301,
        backend="python",max_collisions=100)
    assert r["result"]["outcome_counts"] == plain["outcome_counts"]
    assert r["result"]["estimates"] == _json_data(plain["estimates"])
    assert r["result"]["run_info"]["reference_profile"] == c["reference_profile"]
    assert r["ml_record"]["metadata"]["reference_profile"] == c["reference_profile"]
    assert "reference_profile" not in r["ml_record"]["features"]
    digest = physics.export_slab_experiment(r,tmp_path / "run")
    restored = physics.load_slab_experiment(tmp_path / "run",expected_sha256=digest)
    assert restored == r
    replay = physics.run_slab_experiment(restored["configuration"])
    assert replay["result"]["outcome_counts"] == r["result"]["outcome_counts"]
    assert replay["result"]["estimates"] == r["result"]["estimates"]
    if physics.HAS_NUMBA:
        compiled = physics.simulate_slab([d],[0.,2.],700,seed=301,material_mode="reference",backend="numba",max_collisions=100)
        assert compiled["outcome_counts"] == r["result"]["outcome_counts"]
        assert _json_data(compiled["estimates"]) == r["result"]["estimates"]


def test_profile_mixing_and_library_mixing_are_rejected():
    from material import REFERENCE_POINT_PROFILE, REFERENCE_CONTROL_PROFILE, REFERENCE_MAXWELL_PROFILE
    layers = [graphite_profile(p) for p in (REFERENCE_POINT_PROFILE,REFERENCE_CONTROL_PROFILE,REFERENCE_MAXWELL_PROFILE)]
    for a,b in ((layers[0],layers[1]),(layers[1],layers[2])):
        with pytest.raises(ValueError,match="same energy profile"):
            physics.simulate_slab([a,b],[0.,1.,2.],10,material_mode="reference")
    # Original JENDL-5 water and graphite have matching spectrum/library.
    r = physics.simulate_slab([reference_water(),layers[0]],[0.,.1,.2],10,seed=3,material_mode="reference")
    assert r["run_info"]["reference_profile"]["profile_id"] == REFERENCE_POINT_PROFILE


@pytest.mark.parametrize("field,value", [("profile_id","unknown"),("energy_bounds_ev",[1e-5,1.]),
    ("spectrum_kT_ev",.04),("material_temperature_K",298.15),("spectrum_temperature_K",300.),
    ("numerical_verification_status","scientifically_validated"),("source_library","JENDL-5")])
def test_profile_metadata_forgery_rejected_even_with_recomputed_configuration_hash(tmp_path,field,value):
    from material import REFERENCE_MAXWELL_PROFILE, _content_hash
    c = physics.create_slab_configuration("profile-forgery",[graphite_profile(REFERENCE_MAXWELL_PROFILE)],
        [0.,1.],20,seed=9,material_mode="reference")
    r = physics.run_slab_experiment(c)
    c = r["configuration"]
    c["reference_profile"][field] = value
    c["configuration_id"] = _content_hash({k:v for k,v in c.items() if k != "configuration_id"})
    with pytest.raises(ValueError,match="reference profile mismatch"):
        physics.export_slab_experiment(r,tmp_path / "bad")


@pytest.mark.parametrize("kind", ["reference","legacy"])
def test_original_v2_experiment_archive_remains_readable(tmp_path,kind):
    from material import _content_hash
    layer = reference_water() if kind == "reference" else medium(.3,.2)
    c = physics.create_slab_configuration("v2-compatibility",[layer],[0.,.2],100,seed=71,material_mode=kind)
    c["schema_version"] = "heliosmc.slab-configuration.v2"
    del c["reference_profile"]
    c["configuration_id"] = _content_hash({k:v for k,v in c.items() if k != "configuration_id"})
    r = physics.run_slab_experiment(c)
    assert "reference_profile" not in r["result"]["run_info"]
    assert "reference_profile" not in r["ml_record"]["metadata"]
    digest = physics.export_slab_experiment(r,tmp_path / "v2")
    assert physics.load_slab_experiment(tmp_path / "v2",expected_sha256=digest) == r


def test_fixed_spectrum_group_warning_and_repeat_group_identity():
    from material import REFERENCE_MAXWELL_PROFILE
    ids = []
    for seed in (19,20):
        c = physics.create_slab_configuration("group-repeat",[graphite_profile(REFERENCE_MAXWELL_PROFILE)],
            [0.,1.],100,seed=seed,material_mode="reference")
        r = physics.run_slab_experiment(c)
        ids.append(c["group_id"])
        assert "reference_free_gas_fixed_spectrum_approximation" in r["result"]["warnings"]
        assert "prescribed_flux_spectrum_not_evolved" in r["result"]["limitations"]
        assert not r["ml_record"]["metadata"]["eligible_for_training"]
    assert ids[0] == ids[1]
