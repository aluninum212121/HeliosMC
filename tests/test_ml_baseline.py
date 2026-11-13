"""Small numerical checks; no expensive research run in the unit suite."""
import copy
import hashlib
import json

import numpy as np
import pytest
from scipy.optimize import check_grad

from benchmarks import ml_baseline as ml


@pytest.fixture
def regression_data():
    x = np.linspace(.08, 2.9, 40)[:, None]
    return x, np.exp(-x[:, 0])


@pytest.mark.parametrize("bad", [[[np.nan]], [[np.inf]], [[-.1]], [[3.1]],
                                 [1.], [[1., 2.]], [[True]], [["1"]], []])
def test_invalid_features_rejected(bad):
    with pytest.raises(ValueError):
        ml.inputs(bad)


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-.1], [1.1], [[.3]], [True], []])
def test_invalid_target_rejected(bad):
    with pytest.raises(ValueError):
        ml.targets(bad, 1)


def test_feature_contract_has_only_thickness():
    assert ml.FEATURES == ["thickness_cm"]
    np.testing.assert_array_equal(ml.inputs([[.1], [2.]]), [[.1], [2.]])


def test_group_split_reproducible_with_repeats_and_order():
    groups = [str(i) for i in range(40)]
    split = ml.split_groups(groups)
    assert split == ml.split_groups(groups[::-1]*3)
    assert split != ml.split_groups(groups, seed=11)
    flattened = [g for ids in split.values() for g in ids]
    assert len(flattened) == len(set(flattened)) == 40


@pytest.mark.parametrize("widths", [[1, 3, 1], [1, 3, 2, 1]])
def test_analytic_gradient_matches_finite_difference(widths):
    rng = np.random.default_rng(9)
    theta = rng.normal(0, .2, sum(a*b+b for a,b in zip(widths[:-1],widths[1:])))
    x = np.array([[-1.], [.1], [.8]])
    y = np.array([.1, .5, .7])
    error = check_grad(lambda t: ml.objective(t,widths,x,y,.003)[0],
                       lambda t: ml.objective(t,widths,x,y,.003)[1],theta)
    assert error < 1e-6


@pytest.mark.parametrize("kind", ["dummy", "ridge", "mlp"])
def test_pipeline_scaler_train_only_and_roundtrip(tmp_path, regression_data, kind):
    x,y = regression_data
    train = x[:30],y[:30]
    model = ml.fit_model(*train,kind,validation=(x[30:],y[30:]))
    assert model["success"]
    assert model["mean"] == [float(train[0].mean())]
    assert model["mean"] != [float(x.mean())]
    path=tmp_path/"model.json"
    ml.write_json(path,model)
    loaded=ml.load_model(path)
    np.testing.assert_array_equal(ml.predict(model,x),ml.predict(loaded,x))
    assert np.isfinite(ml.predict(loaded,[[1.77]])).all()
    if kind == "mlp":
        assert ml.metrics(train[1],ml.predict(model,train[0]))["rmse"] < .01
        other=ml.fit_model(*train,kind,validation=(x[30:],y[30:]))
        np.testing.assert_allclose(ml.predict(model,x),ml.predict(other,x),rtol=0,atol=1e-12)


def test_validation_does_not_change_scaler_or_optimization(regression_data):
    x,y=regression_data
    first=ml.fit_model(x[:30],y[:30],validation=(x[30:],y[30:]))
    second=ml.fit_model(x[:30],y[:30],validation=(x[30:],np.zeros(10)))
    np.testing.assert_array_equal(first["theta"],second["theta"])
    assert first["mean"] == second["mean"]


def test_metrics_known_values_and_no_prediction_clipping():
    result=ml.metrics([0.,.5,1.],[.1,.3,1.3])
    assert result["mae"] == pytest.approx(.2)
    assert result["rmse"] == pytest.approx(np.sqrt(.14/3))
    assert result["r2"] == pytest.approx(.72)
    assert result["bias"] == pytest.approx(.2/3)
    assert result["max_absolute_error"] == pytest.approx(.3)
    assert result["outside_fraction"] == 1
    assert ml.metrics([.5,.5],[.4,.6])["r2"] is None
    with pytest.raises(ValueError):
        ml.metrics([.1,.2],[.1])


def test_bad_saved_model_rejected(tmp_path, regression_data):
    model=ml.fit_model(*regression_data,"ridge")
    for i,changed in enumerate((dict(features=["seed"]),dict(scale=[0]),dict(success=False),dict(mean=[float("nan")]))):
        bad={**model,**changed}
        path=tmp_path/f"bad{i}.json"
        path.write_text(json.dumps(bad),encoding="utf-8")
        with pytest.raises(ValueError):
            ml.load_model(path)


def test_real_reference_export_integration_and_unchanged_core(tmp_path):
    names=("physics.py","material.py","numba_compat.py")
    before={n:hashlib.sha256((ml.ROOT/n).read_bytes()).hexdigest() for n in names}
    c=ml.configuration(.5,998,100)
    experiment=ml.physics.run_slab_experiment(c)
    destination=tmp_path/"bundle"
    ml.physics.export_slab_experiment(experiment,destination)
    restored=ml.physics.load_slab_experiment(destination)
    record=ml.observation(restored)
    assert not record["certified_training_eligible"]
    assert record["target"] == restored["result"]["outcome_counts"]["escaped_right"]/100
    assert record["thickness_cm"] == .5
    replay=ml.physics.run_slab_experiment(c)
    assert replay["result"]["outcome_counts"] == experiment["result"]["outcome_counts"]
    assert before == {n:hashlib.sha256((ml.ROOT/n).read_bytes()).hexdigest() for n in names}
    bad=copy.deepcopy(experiment)
    bad["result"]["histories_complete"]=False
    with pytest.raises(ValueError,match="incomplete"):
        ml.observation(bad)
    bad=copy.deepcopy(experiment)
    bad["configuration"]["parameters"]["source_mu"]=.5
    with pytest.raises(ValueError,match="incompatible"):
        ml.observation(bad)
    with pytest.raises(ValueError,match="duplicate"):
        ml.aggregate([record,record])


def test_aggregate_pools_counts_and_separates_replicate_sem():
    rows=[dict(seed=i,group_id="a",thickness_cm=1.,histories=100,
               successes=k,target=k/100) for i,k in enumerate([30,40,50])]
    result=ml.aggregate(rows)[0]
    assert result["target"] == .4
    assert result["standard_error"] == pytest.approx(np.sqrt(.4*.6/300))
    assert result["replicate_sem"] == pytest.approx(.1/np.sqrt(3))


def test_partition_overlap_rejected():
    with pytest.raises(ValueError,match="overlap"):
        ml.partition_arrays([dict(configuration_id="a",thickness_cm=1.,target=.5)],
                            dict(train=["a"],test=["a"],validation=[]))
