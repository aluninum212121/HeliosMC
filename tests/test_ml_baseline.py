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


@pytest.fixture
def multivariate_data():
    x=ml.design_points(60,44)
    # A known small-network response isolates schema/pipeline correctness from
    # optimizing an arbitrary difficult function. Real MC convergence is checked
    # separately in quick/full, with unsuccessful attempts retained.
    y=.45+.2*np.tanh(.5*x[:,0]-.6*x[:,1]+.2*x[:,2])
    return x,y


def test_multivariate_named_feature_order():
    spec=ml.feature_spec()
    x=[[1.,.4,.7]]
    np.testing.assert_array_equal(ml.inputs(x,feature_spec=spec,feature_names=[s["name"] for s in spec]),x)
    with pytest.raises(ValueError,match="order"):
        ml.inputs(x,feature_spec=spec,feature_names=["absorption_rate","thickness_cm","scattering_rate"])
    with pytest.raises(ValueError,match="order"):
        ml.feature_spec(["scattering_rate","thickness_cm"])


@pytest.mark.parametrize("bad", [[[1.,.4]],[[1.,.4,.7,1.]],[["1",.4,.7]],
    [[float("nan"),.4,.7]],[[1.,float("inf"),.7]],[[.09,.4,.7]],[[1.,0.,.7]],
    [[1.,.4,-.1]],[[1.,1.21,.7]],[[1.,.4,2.01]]])
def test_multivariate_invalid_inputs(bad):
    with pytest.raises(ValueError):
        ml.inputs(bad,feature_spec=ml.feature_spec())


@pytest.mark.parametrize("widths", [[3,4,1],[3,4,2,1]])
def test_multivariate_gradient(widths):
    rng=np.random.default_rng(25)
    weights=rng.normal(0,.2,sum(a*b+b for a,b in zip(widths[:-1],widths[1:])))
    x=rng.normal(size=(5,3)); y=rng.uniform(size=5)
    error=check_grad(lambda t:ml.objective(t,widths,x,y,.01)[0],
                     lambda t:ml.objective(t,widths,x,y,.01)[1],weights)
    assert error<1e-6


@pytest.mark.parametrize("kind",["dummy","ridge","polynomial_ridge","mlp"])
def test_multivariate_pipeline_roundtrip_train_scaler(tmp_path,multivariate_data,kind):
    x,y=multivariate_data
    model=ml.fit_model(x[:45],y[:45],kind,feature_spec=ml.feature_spec(),hidden=(8,),
                       validation=(x[45:],y[45:]))
    assert model["schema"]==ml.VERSION_V2
    assert model["success"],model.get("message")
    np.testing.assert_array_equal(model["mean"],x[:45].mean(axis=0))
    assert not np.array_equal(model["mean"],x.mean(axis=0))
    path=tmp_path/(kind+".json"); ml.write_json(path,model)
    np.testing.assert_array_equal(ml.predict(model,x),ml.predict(ml.load_model(path),x))
    assert ml.predict(model,x[:3]).shape==(3,)


def test_polynomial_interactions_and_constant_feature(multivariate_data):
    x,_=multivariate_data; x[:,2]=.7
    y=.1+.03*x[:,0]+.02*x[:,1]**2+.04*x[:,0]*x[:,1]
    model=ml.fit_model(x,y,"polynomial_ridge",alpha=0.,feature_spec=ml.feature_spec())
    assert model["constant_features"]==[False,False,True]
    assert model["scale"][2]==1.
    np.testing.assert_allclose(ml.predict(model,x),y,atol=1e-12)
    assert len(model["coefficients"])==10


@pytest.mark.parametrize("change",["order","domain","scale","theta","widths","coefficients","nonfinite_bias"])
def test_v2_model_schema_tampering(tmp_path,multivariate_data,change):
    x,y=multivariate_data
    kind="ridge" if change=="coefficients" else "mlp"
    model=ml.fit_model(x,y,kind,feature_spec=ml.feature_spec(),hidden=(4,))
    model["success"]=True  # Loader validation is independent of optimization on this tiny fixture.
    if change=="order": model["features"]=model["features"][::-1]
    if change=="domain": model["feature_spec"][0]["domain"]=[0.,100.]
    if change=="scale": model["scale"]=[1.]
    if change=="theta": model["theta"]=model["theta"][:-1]
    if change=="widths": model["widths"][0]=1
    if change=="coefficients": model["coefficients"]=[1.,2.]
    if change=="nonfinite_bias": model["theta"][-1]=float("inf")
    path=tmp_path/"bad.json"; path.write_text(json.dumps(model),encoding="utf-8")
    with pytest.raises(ValueError): ml.load_model(path)


def test_v1_saved_model_still_predicts():
    path=ml.ROOT/"benchmarks/results/ml_baseline/full/mlp.json"
    model=ml.load_model(path)
    assert ml.predict(model,[[1.5]])[0]==pytest.approx(.32972161544684164,abs=1e-14)
    with pytest.raises(ValueError): ml.predict(model,[[1.5,.5,1.]])


def test_lhs_reproducible_covers_each_marginal():
    points=ml.design_points(60,13)
    np.testing.assert_array_equal(points,ml.design_points(60,13))
    assert not np.array_equal(points,ml.design_points(60,14))
    for i,spec in enumerate(ml.feature_spec()):
        lo,hi=spec["domain"]
        assert len(set(np.floor((points[:,i]-lo)/(hi-lo)*60)))==60


def test_v2_native_generation_resume_and_aggregation(tmp_path):
    plan=dict(version=ml.VERSION_V2,groups=10,repeats=3,histories_per_repeat=50,quick=True,budget_seconds=300.)
    config,records,grouped=ml.generate(tmp_path,plan)
    assert len(records)==30 and len(grouped)==10
    assert all(r["histories"]==150 and r["repeats"]==3 for r in grouped)
    archive_hash=hashlib.sha256((tmp_path/"experiments.zip").read_bytes()).hexdigest()
    other,other_records,other_grouped=ml.generate(tmp_path,plan)
    assert config==other and records==other_records and grouped==other_grouped
    assert archive_hash==hashlib.sha256((tmp_path/"experiments.zip").read_bytes()).hexdigest()
    arrays=ml.partition_arrays(grouped,config["partitions"],feature_names=config["feature_names"])
    assert arrays["train"][0].shape[1]==3
    assert set(sum(config["partitions"].values(),[]))=={r["configuration_id"] for r in grouped}
    assert len(set(r["seed"] for r in records))==30


def test_group_changes_with_any_v2_input_and_not_seed():
    original=ml.configuration(None,1,20,feature_values=[1.,.4,.8])
    repeat=ml.configuration(None,2,40,feature_values=[1.,.4,.8])
    assert original["group_id"]==repeat["group_id"]
    for point in ([1.1,.4,.8],[1.,.5,.8],[1.,.4,.9]):
        assert ml.configuration(None,1,20,feature_values=point)["group_id"]!=original["group_id"]


def test_v2_serialized_models_cannot_change_feature_units(tmp_path,multivariate_data):
    x,y=multivariate_data
    model=ml.fit_model(x,y,"ridge",feature_spec=ml.feature_spec())
    model["feature_spec"][1]["unit"]="barn"
    path=tmp_path/"bad.json"; ml.write_json(path,model)
    with pytest.raises(ValueError,match="contract"): ml.load_model(path)


def test_cli_v2_predict_requires_all_named_parameters(tmp_path,multivariate_data,capsys,monkeypatch):
    import sys
    x,y=multivariate_data
    model=ml.fit_model(x,y,"ridge",feature_spec=ml.feature_spec())
    ml.write_json(tmp_path/"mlp.json",model)
    argv=["ml_baseline","predict","--output",str(tmp_path),"--thickness","1."]
    monkeypatch.setattr(sys,"argv",argv)
    with pytest.raises(SystemExit): ml.main()
    monkeypatch.setattr(sys,"argv",argv+["--absorption-rate",".4","--scattering-rate",".8"])
    ml.main(); result=json.loads(capsys.readouterr().out)
    assert result["schema"]==ml.VERSION_V2
    assert result["inputs"]["absorption_rate"]==.4


def test_metric_out_of_range_fraction():
    result=ml.metrics([.1,.5,.9],[-.1,.5,1.1])
    assert result["out_of_range_fraction"]==pytest.approx(2/3)


def test_training_does_not_inspect_test_labels(tmp_path,multivariate_data,monkeypatch):
    x,y=multivariate_data
    monkeypatch.setattr(ml,"CANDIDATES_V2",(((4,),1e-6),))
    config=dict(version=ml.VERSION_V2,ridge_alphas=[.01],maxiter=2500,source_hashes=ml.hashes(),
                partitions=dict(train=[str(i) for i in range(45)],validation=[str(i) for i in range(45,60)]),environment={})
    models,selection=ml.train(tmp_path,dict(train=(x[:45],y[:45]),validation=(x[45:],y[45:]),test=object()),config)
    assert not selection["test_used_for_selection"]
    assert models["mlp"]["widths"][0]==3


def test_empty_feature_schema_rejected(tmp_path,multivariate_data):
    x,y=multivariate_data
    model=ml.fit_model(x,y,"ridge",feature_spec=ml.feature_spec())
    model["features"]=[]
    path=tmp_path/"bad.json"; ml.write_json(path,model)
    with pytest.raises(ValueError): ml.load_model(path)


def test_bootstrap_is_paired_and_deterministic():
    y=np.linspace(.1,.9,20)
    predictions=dict(mlp=y+.01,ridge=y+.05,polynomial_ridge=y+.03)
    first=ml.bootstrap_scores(y,predictions,repetitions=100)
    assert first==ml.bootstrap_scores(y,predictions,repetitions=100)
    assert first["intervals"]["mlp_minus_ridge"]==pytest.approx([-.04,-.04])


def test_report_consistency_rejects_changed_table(tmp_path):
    tables={"1":dict(title="fixture",source="fixture.json",rows=[dict(value=.123)])}
    ml.write_json(tmp_path/"report_tables.json",tables)
    path=tmp_path/"report.md"
    path.write_text("<!-- TABLE_1_BEGIN -->\n| value |\n|---|\n| .5 |\n<!-- TABLE_1_END -->",encoding="utf-8")
    with pytest.raises(ValueError,match="report table mismatch"):
        ml.verify_report_v2(tmp_path,path)
