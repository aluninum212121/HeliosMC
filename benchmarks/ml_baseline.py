"""CPU-only v1 water and v2 synthetic multivariate slab surrogate studies.

Run as ``python -m benchmarks.ml_baseline quick|full|evaluate|predict|report``.
This is simulator emulation, not the physics module's certified training subset.
The reference archive's eligible_for_training=False is deliberately preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import tempfile
import zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.stats import qmc

import physics
from material import (REFERENCE_APPROXIMATION, REFERENCE_POINT_PROFILE,
                      get_reference_transport_dataset, material_to_dict,
                      MaterialDataset, MaterialQuantity, resolve_medium_rates,
                      material_from_dict)

VERSION = "water_slab_surrogate_v1"
ROOT = Path(__file__).resolve().parents[1]
DOMAIN = (0.05, 3.0)  # cm; fixed normal incidence, source at the left face
FEATURES = ["thickness_cm"]
TARGET = "escaped_right_fraction"
SEEDS = (17, 29, 43)  # fixed before observing validation or test targets
CANDIDATES = (((8,), 1e-6), ((16,), 1e-6), ((16, 8), 1e-6), ((16,), 1e-4))
VERSION_V2 = "synthetic_slab_surrogate_v2"
# Fixed BEFORE the pilot and any validation/test results. No material identity
# is implied by these invented cm^-1 rates. Maximum total optical width: 9.6.
FEATURE_SPEC_V2 = (
    dict(name="thickness_cm", label="Толщина слоя", unit="cm", quantity_type="length",
         domain=[.1, 3.], position=0, used=True, validation="finite real; closed interval"),
    dict(name="absorption_rate", label="Коэффициент поглощения", unit="1/cm", quantity_type="rate",
         domain=[.02, 1.2], position=1, used=True, validation="finite real; closed interval"),
    dict(name="scattering_rate", label="Коэффициент рассеяния", unit="1/cm", quantity_type="rate",
         domain=[0., 2.], position=2, used=True, validation="finite real; closed interval"),
)
CANDIDATES_V2 = (((16,), 1e-6), ((32,), 1e-6), ((32, 16), 1e-6), ((32, 16), 1e-4))
V2_DESIGN_SEED = 20261119


def feature_spec(names=None):
    """Return an owned, ordered schema, also for reduced ablation estimators."""
    if names is None:
        names = [s["name"] for s in FEATURE_SPEC_V2]
    if not isinstance(names, (tuple, list)) or not names:
        raise ValueError("feature names must be a nonempty ordered sequence")
    if len(names) != len(set(names)) or any(n not in [s["name"] for s in FEATURE_SPEC_V2] for n in names):
        raise ValueError("unknown or duplicate feature")
    chosen = [s for s in FEATURE_SPEC_V2 if s["name"] in names]
    if [s["name"] for s in chosen] != list(names):
        raise ValueError("incorrect feature order")
    return [dict(s, domain=list(s["domain"]), position=i) for i, s in enumerate(chosen)]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("physics.py", "material.py", "numba_compat.py", "benchmarks/ml_baseline.py")}


def water():
    return get_reference_transport_dataset("Вода (H₂O)",
        approximation=REFERENCE_APPROXIMATION, profile_id=REFERENCE_POINT_PROFILE)


def inputs(values, domain=DOMAIN, *, feature_spec=None, feature_names=None):
    """Validate (n, p) against the estimator's schema; v1 remains explicit.

    For bare numeric matrices the caller asserts column order. Named callers
    must supply feature_names; permutations are rejected, never guessed.
    """
    raw = np.asanyarray(values)
    if np.ma.isMaskedArray(raw) or raw.dtype.kind not in "fiu":
        raise ValueError("features must be real numeric values without masks")
    x = np.asarray(raw, dtype=float)
    n_features = len(feature_spec) if feature_spec is not None else 1
    if x.ndim != 2 or x.shape[1] != n_features or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"features must have finite shape (n, {n_features})")
    if feature_spec is not None:
        names = [s["name"] for s in feature_spec]
        if feature_names is not None and list(feature_names) != names:
            raise ValueError("incorrect feature order")
        bounds = np.asarray([s["domain"] for s in feature_spec])
        if np.any((x < bounds[:, 0]) | (x > bounds[:, 1])):
            raise ValueError("input outside model feature domains")
        return x
    if domain is not None and np.any((x < domain[0]) | (x > domain[1])):
        raise ValueError(f"thickness outside trained domain {domain} cm")
    return x


def targets(values, n):
    y = np.asarray(values)
    if y.dtype.kind not in "fiu" or y.shape != (n,) or not np.isfinite(y).all():
        raise ValueError("target must be a finite vector aligned with features")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("transmission fraction outside [0, 1]")
    return y.astype(float)


def split_groups(group_ids, seed=20261113):
    """Split distinct physical groups before any target-dependent selection."""
    groups = np.asarray(sorted(set(group_ids)))
    if len(groups) < 10:
        raise ValueError("at least ten independent configurations required")
    shuffled = np.random.default_rng(seed).permutation(groups)
    n_hold = max(1, round(len(groups) * .15))
    partitions = dict(test=shuffled[:n_hold].tolist(),
                      validation=shuffled[n_hold:2*n_hold].tolist(),
                      train=shuffled[2*n_hold:].tolist())
    sets = [set(partitions[k]) for k in ("train", "validation", "test")]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("physical group leakage")
    return partitions


def metrics(actual, predicted):
    if not len(actual):
        raise ValueError("metrics need at least one configuration")
    actual = targets(actual, len(actual))
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError("invalid prediction vector")
    error = predicted - actual
    ss = float(np.sum((actual - actual.mean())**2))
    return dict(mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                r2=float(1 - np.sum(error**2)/ss) if ss > 0 else None,
                max_absolute_error=float(np.max(abs(error))), bias=float(np.mean(error)),
                outside_fraction=int(np.count_nonzero((predicted < 0) | (predicted > 1))),
                out_of_range_fraction=float(np.mean((predicted < 0) | (predicted > 1))))


def unpack(theta, widths):
    theta = np.asarray(theta)
    if (theta.ndim != 1 or theta.dtype.kind not in "fiu" or not np.isfinite(theta).all() or len(widths) < 2
            or any(type(n) is not int or n < 1 for n in widths)):
        raise ValueError("invalid architecture or weights")
    layers, offset = [], 0
    for left, right in zip(widths[:-1], widths[1:]):
        end = offset + left * right
        layers.append((theta[offset:end].reshape(left, right), theta[end:end+right]))
        offset = end + right
    if offset != len(theta):
        raise ValueError("weight count does not match architecture")
    return layers


def forward(theta, widths, x):
    activations = [x]
    layers = unpack(theta, widths)
    for i, (weight, bias) in enumerate(layers):
        z = activations[-1] @ weight + bias
        activations.append(np.tanh(z) if i < len(layers)-1 else z)
    return activations[-1][:, 0], activations, layers


def objective(theta, widths, x, y, alpha):
    """Half MSE + alpha/2 sum(W^2); analytic backprop, biases unpenalized."""
    prediction, activations, layers = forward(theta, widths, x)
    residual = prediction - y
    loss = .5 * np.mean(residual**2) + .5 * alpha * sum(np.sum(w*w) for w, _ in layers)
    delta = residual[:, None] / len(y)
    gradients = []
    for i in reversed(range(len(layers))):
        weight, _ = layers[i]
        gradients.append((activations[i].T @ delta + alpha * weight, delta.sum(axis=0)))
        if i:
            delta = (delta @ weight.T) * (1 - activations[i]**2)
    grad = np.concatenate([part.ravel() for pair in reversed(gradients) for part in pair])
    return float(loss), grad


def fit_model(x, y, kind="mlp", *, hidden=(8,), alpha=1e-6, seed=17,
              validation=None, maxiter=2000, feature_spec=None):
    """Scaler and estimator are one artifact; fit sees training data only."""
    x = inputs(x, feature_spec=feature_spec)
    y = targets(y, len(x))
    start = perf_counter()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    constant = np.ptp(x, axis=0) == 0
    scale = np.where((scale > 0) & ~constant, scale, 1.)
    z = (x - mean) / scale
    preprocessing = perf_counter() - start
    model = dict(schema=VERSION, kind=kind, features=FEATURES, target=TARGET,
                 domain=list(DOMAIN), mean=mean.tolist(), scale=scale.tolist(),
                 alpha=alpha, seed=seed, preprocessing_seconds=preprocessing,
                 train_min=float(x.min()), train_max=float(x.max()))
    if feature_spec is not None:
        model.update(schema=VERSION_V2, feature_spec=feature_spec,
                     features=[s["name"] for s in feature_spec],
                     domain=[s["domain"] for s in feature_spec],
                     train_min=x.min(axis=0).tolist(), train_max=x.max(axis=0).tolist(),
                     constant_features=constant.tolist())
    start = perf_counter()
    if kind == "dummy":
        model.update(value=float(y.mean()), success=True, iterations=0)
    elif kind in ("ridge", "polynomial_ridge"):
        design = polynomial_design(z) if kind == "polynomial_ridge" else np.column_stack((z, np.ones(len(z))))
        penalty = np.diag([alpha] * (design.shape[1]-1) + [0.])
        # Augmented least squares avoids singular normal equations (constant or
        # dependent features); the final intercept is intentionally unpenalized.
        coefficients = np.linalg.lstsq(np.vstack((design, np.sqrt(penalty))),
            np.concatenate((y, np.zeros(design.shape[1]))), rcond=None)[0]
        model.update(coefficients=coefficients.tolist(), success=True, iterations=1)
        if kind == "polynomial_ridge":
            model["degree"] = 2
    elif kind == "mlp":
        widths = [x.shape[1], *hidden, 1]
        rng = np.random.default_rng(seed)
        theta = np.concatenate([part.ravel() for left, right in zip(widths[:-1], widths[1:])
            for part in (rng.uniform(-np.sqrt(6/(left+right)), np.sqrt(6/(left+right)), (left, right)),
                         np.zeros(right))])
        history = []
        if validation is not None:
            vx = (inputs(validation[0], feature_spec=feature_spec) - mean) / scale
            vy = targets(validation[1], len(vx))

        def record(weights):
            loss, _ = objective(weights, widths, z, y, alpha)
            row = dict(iteration=len(history), penalized_train_loss=loss,
                       train_rmse=metrics(y, forward(weights, widths, z)[0])["rmse"])
            if validation is not None:
                row["validation_rmse"] = metrics(vy, forward(weights, widths, vx)[0])["rmse"]
            history.append(row)

        record(theta)
        options = dict(maxiter=maxiter, maxfun=maxiter*3, ftol=1e-12, gtol=1e-7, maxls=40)
        if feature_spec is not None:
            # These tolerances are below the planned MC label noise. Retaining
            # v1's tighter ftol needlessly exhausts iterations on reduced models.
            options.update(ftol=1e-10, gtol=1e-6)
        result = minimize(objective, theta, args=(widths, z, y, alpha), jac=True,
                          method="L-BFGS-B", callback=record, options=options)
        model.update(widths=widths, theta=result.x.tolist(), history=history,
                     optimizer="L-BFGS-B", options=options, activation="tanh", output="linear",
                     success=bool(result.success), status=int(result.status),
                     message=str(result.message), iterations=int(result.nit),
                     gradient_max=float(np.max(abs(result.jac))), objective=float(result.fun))
    else:
        raise ValueError("unknown estimator")
    model["training_seconds"] = perf_counter() - start
    return model


def polynomial_design(z):
    """[z_i, z_i*z_j for i<=j, 1]; 10 columns for three original inputs."""
    return np.column_stack([z] + [z[:, i]*z[:, j] for i in range(z.shape[1])
                                                for j in range(i, z.shape[1])] + [np.ones(len(z))])


def predict(model, x, *, feature_names=None):
    x = inputs(x, model["domain"], feature_spec=model.get("feature_spec"), feature_names=feature_names)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    if model["kind"] == "dummy":
        return np.full(len(x), model["value"])
    if model["kind"] in ("ridge", "polynomial_ridge"):
        design = polynomial_design(z) if model["kind"] == "polynomial_ridge" else np.column_stack((z, np.ones(len(z))))
        return design @ np.asarray(model["coefficients"])
    if model["kind"] == "mlp":
        return forward(np.asarray(model["theta"]), model["widths"], z)[0]
    raise ValueError("unknown estimator")


def load_model(path):
    """JSON numeric parameters only: no pickle or executable deserialization."""
    if Path(path).stat().st_size > 10_000_000:
        raise ValueError("model file too large")
    model = read_json(path)
    v2 = model.get("schema") == VERSION_V2
    if v2:
        expected = feature_spec(model.get("features"))
        if model.get("feature_spec") != expected or model.get("domain") != [s["domain"] for s in expected]:
            raise ValueError("unsupported feature contract")
        p = len(expected)
        middle = [[sum(s["domain"])/2 for s in expected]]
    else:
        p, middle = 1, [[sum(DOMAIN)/2]]
    if ((not v2 and (model.get("schema") != VERSION or model.get("features") != FEATURES
                    or model.get("domain") != list(DOMAIN)))
            or model.get("target") != TARGET
            or model.get("kind") not in ("dummy", "ridge", "polynomial_ridge", "mlp")):
        raise ValueError("unsupported model schema or feature contract")
    for name in ("mean", "scale"):
        a = np.asarray(model[name], dtype=float)
        if a.shape != (p,) or not np.isfinite(a).all():
            raise ValueError("invalid scaler")
    if np.any(np.asarray(model["scale"]) <= 0) or not model.get("success"):
        raise ValueError("invalid scaler or unconverged model")
    if model["kind"] == "mlp":
        widths = model["widths"]
        if (len(widths) not in (3, 4) or widths[0] != p or widths[-1] != 1
                or any(type(n) is not int or not 1 <= n <= 64 for n in widths)):
            raise ValueError("unsupported architecture")
        unpack(np.asarray(model["theta"]), widths)
    elif model["kind"] in ("ridge", "polynomial_ridge"):
        count = p+1 if model["kind"] == "ridge" else 1+p+p*(p+1)//2
        coefficients = np.asarray(model["coefficients"])
        if coefficients.shape != (count,) or not np.isfinite(coefficients).all():
            raise ValueError("invalid regression coefficients")
    if not np.isfinite(predict(model, middle)).all():
        raise ValueError("nonfinite model")
    return model


def synthetic_medium(absorption, scattering):
    return MaterialDataset("heliosmc:ml-synthetic-v2", "Abstract educational medium", "synthetic",
        "Invented constant rates for simulator emulation; not measured nuclear data", "2",
        ("Nonreactive, isotropic, fixed one-group closure; no real substance identity.",),
        (MaterialQuantity("absorption_rate", "1/cm", float(absorption)),
         MaterialQuantity("scattering_rate", "1/cm", float(scattering))))


def configuration(thickness, seed, histories, *, feature_values=None):
    if feature_values is not None:
        x = inputs([feature_values], feature_spec=feature_spec())[0]
        return physics.create_slab_configuration("ml-synthetic-v2", [synthetic_medium(x[1], x[2])],
            [0., float(x[0])], histories, seed=int(seed), max_collisions=1000, max_samples=0,
            source_mu=1., source_x=0., material_mode="legacy", backend="auto")
    return physics.create_slab_configuration("ml-reference-water", [water()], [0., float(thickness)],
        histories, seed=int(seed), max_collisions=1000, max_samples=0, source_mu=1.,
        source_x=0., material_mode="reference", backend="auto")


def observation(experiment):
    """Explicit emulation QC; never changes the archive's certification flag."""
    c, r = experiment["configuration"], experiment["result"]
    p = c["parameters"]
    if c["scenario_id"] == "ml-synthetic-v2":
        if (c["material_mode"] != "legacy" or len(c["layers"]) != 1
                or p["source_x"] != 0. or p["source_mu"] != 1.
                or len(p["edges_cm"]) != 2 or p["edges_cm"][0] != 0.
                or c["algorithm_version"] != "abstract_slab_v1"):
            raise ValueError("incompatible synthetic scenario")
        layer = material_from_dict(c["layers"][0])
        a, s = resolve_medium_rates(layer)
        if c["layers"][0] != material_to_dict(synthetic_medium(a, s)):
            raise ValueError("incompatible synthetic provenance")
        x = inputs([[p["edges_cm"][1], a, s]], feature_spec=feature_spec())
        extra = dict(absorption_rate=a, scattering_rate=s, dataset_schema=VERSION_V2,
                     algorithm_version=c["algorithm_version"], warnings=r["warnings"],
                     code_sha256=c["code_sha256"], material_kind=layer.data_kind)
    else:
        extra = {}
    if not extra and (c["material_mode"] != "reference" or c["reference_profile"]["profile_id"] != REFERENCE_POINT_PROFILE
            or c["layers"] != [material_to_dict(water())] or p["source_x"] != 0.
            or p["source_mu"] != 1. or len(p["edges_cm"]) != 2 or p["edges_cm"][0] != 0.):
        raise ValueError("incompatible material, profile, geometry or source")
    if not extra:
        x = inputs([[p["edges_cm"][1]]])
    if (r["execution_status"] != "completed" or not r["histories_complete"]
            or r["unfinished_histories"] or r["processed_histories"] != p["n_histories"]
            or p["n_histories"] <= 0):
        raise ValueError("incomplete or empty simulation")
    counts = r["outcome_counts"]
    if sum(counts.values()) != p["n_histories"]:
        raise ValueError("outcome balance mismatch")
    estimate = r["estimates"]["escaped_right"]
    value = targets([estimate["estimate"]], 1)[0]
    if value != counts["escaped_right"] / p["n_histories"]:
        raise ValueError("target/count mismatch")
    se = estimate["standard_error"]
    if se is None or not np.isfinite(se) or se < 0:
        raise ValueError("missing sampling uncertainty")
    return dict(group_id=c["group_id"], run_configuration_id=c["configuration_id"],
                thickness_cm=float(x[0, 0]), seed=p["seed"], target=float(value),
                successes=counts["escaped_right"], histories=p["n_histories"], standard_error=se,
                seconds=r["timing"]["total_seconds"], transport_seconds=r["timing"]["transport_seconds"],
                ci_low=estimate["confidence_interval"][0], ci_high=estimate["confidence_interval"][1],
                status=r["execution_status"], **extra, certified_training_eligible=
                experiment["ml_record"]["metadata"]["eligible_for_training"])


def aggregate(records):
    groups = {}
    seen_seeds = set()
    for row in records:
        # Independent streams across the whole dataset, not just within a group.
        if row["seed"] in seen_seeds:
            raise ValueError("duplicate Monte Carlo seed")
        seen_seeds.add(row["seed"])
        groups.setdefault(row["group_id"], []).append(row)
    result = []
    for group, rows in sorted(groups.items()):
        names = [s["name"] for s in FEATURE_SPEC_V2] if "absorption_rate" in rows[0] else FEATURES
        if len({tuple(r[n] for n in names) for r in rows}) != 1:
            raise ValueError("group feature mismatch")
        n = sum(r["histories"] for r in rows)
        k = sum(r["successes"] for r in rows)
        stats = physics.summarize_binomial(k, n)
        # Pooled Bernoulli SE and independent-replicate SEM are distinct diagnostics.
        replicate_sem = float(np.std([r["target"] for r in rows], ddof=1)/np.sqrt(len(rows))) if len(rows)>1 else None
        result.append(dict(configuration_id=group, **{n: rows[0][n] for n in names},
            target=stats["estimate"], standard_error=stats["standard_error"],
            ci_low=stats["confidence_interval"][0], ci_high=stats["confidence_interval"][1],
            replicate_sem=replicate_sem, repeats=len(rows), histories=n, successes=k))
    return result


def save_csv(path, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_bundle(archive, prefix, directory):
    directory.mkdir()
    # Fixed members only, no untrusted ZIP extraction paths.
    for name in ("experiment.json", "results.csv", "arrays.npz"):
        entry = archive.getinfo(prefix + "/" + name)
        if entry.file_size > 10_000_000:
            raise ValueError("oversized archive member")
        (directory / name).write_bytes(archive.read(entry))
    return physics.load_slab_experiment(directory)


def load_dataset(directory):
    rows = []
    with zipfile.ZipFile(Path(directory)/"experiments.zip") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member")
        prefixes = sorted({n.rsplit("/", 1)[0] for n in names})
        if len(names) != 3 * len(prefixes):
            raise ValueError("unexpected bundle members")
        with tempfile.TemporaryDirectory(prefix="heliosmc-ml-read-") as tmp:
            for i, prefix in enumerate(prefixes):
                experiment = read_bundle(archive, prefix, Path(tmp)/str(i))
                rows.append(observation(experiment))
    return rows, aggregate(rows)


def pilot(directory, quick, *, version=VERSION, budget_seconds=300.):
    """A real small experiment precedes selection of the main compute budget."""
    if version == VERSION_V2:
        return pilot_v2(directory, quick, budget_seconds)
    pilot_path = directory / "pilot.json"
    if pilot_path.exists():
        return read_json(pilot_path)
    rows, first = [], None
    for i, width in enumerate(np.linspace(*DOMAIN, 5)):
        for repeat in range(3):
            c = configuration(width, 71000+i*3+repeat, 2000)
            start = perf_counter()
            experiment = physics.run_slab_experiment(c)
            elapsed = perf_counter()-start
            if first is None:
                first = dict(wrapper_seconds=elapsed, **experiment["result"]["timing"])
            row = observation(experiment)
            row["wrapper_seconds"] = elapsed
            rows.append(row)
    # Planning is based only on pilot cost, never on held-out accuracy.
    cost = float(np.median([r["wrapper_seconds"] for r in rows[1:]]))
    groups = 30 if quick else int(np.clip(180./(3*cost*10), 60, 180))
    plan = dict(version=VERSION, first_call_including_possible_jit=first, records=rows,
                warm_wrapper_median_seconds=cost, groups=groups, repeats=3,
                histories_per_repeat=2000 if quick else 20000,
                planning_budget_seconds=180, quick=quick,
                target_range=[min(r["target"] for r in rows), max(r["target"] for r in rows)],
                source_hashes=hashes())
    if np.ptp([r["target"] for r in rows]) <= .05:
        raise RuntimeError("pilot target has insufficient variation")
    write_json(pilot_path, plan)
    print(f"Pilot: {groups} groups, 3 repeats, {plan['histories_per_repeat']} histories/repeat", flush=True)
    return plan


def generate(directory, plan):
    if plan.get("version") == VERSION_V2:
        return generate_v2(directory, plan)
    config_path = directory / "experiment_config.json"
    if config_path.exists():
        config = read_json(config_path)
        if config["source_hashes"] != hashes() or config["quick"] != plan["quick"]:
            raise ValueError("resume requires identical code and experiment mode; use a new output directory")
    else:
        rng = np.random.default_rng(20261113)
        widths = DOMAIN[0] + (DOMAIN[1]-DOMAIN[0])*(np.arange(plan["groups"])+rng.random(plan["groups"]))/plan["groups"]
        configs = [configuration(w, 100000+i*plan["repeats"], plan["histories_per_repeat"])
                   for i, w in enumerate(widths)]
        partitions = split_groups([c["group_id"] for c in configs])
        # The narrow advertised domain is bounded by observed training inputs.
        # Swap extreme sampled configurations into training using inputs only.
        for c in (configs[0], configs[-1]):
            gid = c["group_id"]
            for name in ("validation", "test"):
                if gid in partitions[name]:
                    swap = next(g for g in partitions["train"] if g not in (configs[0]["group_id"], configs[-1]["group_id"]))
                    partitions[name].remove(gid)
                    partitions[name].append(swap)
                    partitions["train"].remove(swap)
                    partitions["train"].append(gid)
        config = dict(version=VERSION, source_hashes=hashes(), quick=plan["quick"],
            widths_cm=widths.tolist(), groups=[c["group_id"] for c in configs],
            partitions=partitions, repeats=plan["repeats"], histories_per_repeat=plan["histories_per_repeat"],
            seed_base=100000, design_seed=20261113, split_seed=20261113, training_seeds=list(SEEDS),
            candidates=[dict(hidden=list(h), alpha=a) for h,a in CANDIDATES],
            selection="minimum mean validation RMSE across fixed training seeds; primary model always seed 17",
            accuracy_criterion="test RMSE <= 0.01 absolute fraction (predeclared educational tolerance)",
            features=FEATURES, target=TARGET, domain_cm=list(DOMAIN),
            physical_template=configs[0], scientific_validation="not_established",
            reference_training_exclusion="preserved; separate research emulation QC, not certified training",
            analytic_baseline="not applicable: exp(-Sigma_t*d) is uncollided transmission, not total right escape with scattering",
            environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                numba=getattr(sys.modules.get("numba"), "__version__", None), platform=platform.platform(),
                processor=platform.processor()),
            reproduction=f"python -B -m benchmarks.ml_baseline {'quick' if plan['quick'] else 'full'} --output <new-directory>")
        write_json(config_path, config)
    archive_path = directory / "experiments.zip"
    started = perf_counter()
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing = set(archive.namelist())
        for i, width in enumerate(config["widths_cm"]):
            for repeat in range(config["repeats"]):
                prefix = f"g{i:04d}_r{repeat}"
                members = {prefix+"/"+n for n in ("experiment.json", "results.csv", "arrays.npz")}
                if members <= existing:
                    continue
                if members & existing:
                    raise ValueError("incomplete archived bundle; preserve this output and use a new directory")
                c = configuration(width, config["seed_base"]+i*config["repeats"]+repeat, config["histories_per_repeat"])
                try:
                    experiment = physics.run_slab_experiment(c)
                    observation(experiment)
                except Exception as exc:
                    failure = directory / f"failure_{prefix}.json"
                    if not failure.exists():
                        write_json(failure, dict(configuration=c, reason=str(exc)))
                    raise
                with tempfile.TemporaryDirectory(prefix="heliosmc-ml-export-") as tmp:
                    bundle = Path(tmp)/"bundle"
                    physics.export_slab_experiment(experiment, bundle)
                    restored = physics.load_slab_experiment(bundle)
                    if restored != experiment:
                        raise RuntimeError("export roundtrip mismatch")
                    for item in sorted(bundle.iterdir()):
                        archive.write(item, prefix+"/"+item.name)
            if (i+1) % 30 == 0:
                print(f"Archived {i+1}/{len(config['widths_cm'])} configurations", flush=True)
    elapsed = perf_counter()-started
    rows, grouped = load_dataset(directory)
    if len(grouped) != len(config["groups"]) or any(r["repeats"] != config["repeats"] for r in grouped):
        raise ValueError("incomplete dataset")
    if not (directory/"generation.json").exists():
        write_json(directory/"generation.json", dict(current_generation_wall_seconds=elapsed,
            wall_scope="current generation invocation incl export/reload; on resume may omit previous invocation",
            simulation_seconds=sum(r["seconds"] for r in rows),
            transport_seconds=sum(r["transport_seconds"] for r in rows),
            excluded_records=[], records=len(rows), groups=len(grouped)))
    if not (directory/"dataset.csv").exists():
        save_csv(directory/"dataset.csv", grouped)
    return config, rows, grouped


def partition_arrays(grouped, partitions, *, feature_names=None):
    names = feature_names or FEATURES
    by_group = {r["configuration_id"]: r for r in grouped}
    flattened = [g for ids in partitions.values() for g in ids]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(by_group):
        raise ValueError("split overlap or missing groups")
    return {name: (np.array([[by_group[g][n] for n in names] for g in ids]),
                   np.array([by_group[g]["target"] for g in ids])) for name, ids in partitions.items()}


def train(directory, arrays, config):
    if config.get("version") == VERSION_V2:
        return train_v2(directory, arrays, config)
    path = directory/"selection.json"
    if path.exists():
        selection = read_json(path)
        return {k: load_model(directory/(k+".json")) for k in ("dummy", "ridge", "mlp")}, selection
    x, y = arrays["train"]
    vx, vy = arrays["validation"]
    dummy = fit_model(x, y, "dummy")
    ridge_candidates = [fit_model(x, y, "ridge", alpha=a) for a in (0., .01, 1.)]
    ridge = min(ridge_candidates, key=lambda m: metrics(vy, predict(m, vx))["rmse"])
    trials, candidates = [], []
    for hidden, alpha in CANDIDATES:
        models = []
        for seed in SEEDS:
            m = fit_model(x, y, hidden=hidden, alpha=alpha, seed=seed, validation=(vx, vy))
            models.append(m)
            trials.append(dict(hidden=list(hidden), alpha=alpha, seed=seed, success=m["success"],
                message=m["message"], iterations=m["iterations"], gradient_max=m["gradient_max"],
                training_seconds=m["training_seconds"], train=metrics(y, predict(m, x)),
                validation=metrics(vy, predict(m, vx)), history=m["history"]))
        candidates.append((float(np.mean([metrics(vy, predict(m, vx))["rmse"] for m in models])), models))
        print(f"MLP {hidden}, alpha={alpha}: validation RMSE {candidates[-1][0]:.6f}; converged {[m['success'] for m in models]}", flush=True)
    valid = [c for c in candidates if all(m["success"] for m in c[1])]
    if not valid:
        write_json(directory/"training_failure.json", dict(trials=trials, reason="no fully converged candidate"))
        raise RuntimeError("no converged candidate; diagnostics saved")
    _, selected = min(valid, key=lambda c: c[0])
    model = selected[0]  # Never select the luckiest random seed.
    model["experiment"] = dict(source_hashes=config["source_hashes"], dataset_version=VERSION,
        training_groups=config["partitions"]["train"], validation_groups=config["partitions"]["validation"],
        environment=config["environment"], profile=REFERENCE_POINT_PROFILE,
        scientific_validation="not_established", status="optimizer_converged")
    # Genuine repeated training with identical inputs/settings, before test evaluation.
    repeated = fit_model(x, y, hidden=tuple(model["widths"][1:-1]), alpha=model["alpha"],
                         seed=model["seed"], validation=(vx, vy))
    reproduction_difference = float(np.max(abs(predict(model, vx)-predict(repeated, vx))))
    if not repeated["success"] or reproduction_difference > 1e-10:
        raise RuntimeError("training reproducibility check failed")
    models = dict(dummy=dummy, ridge=ridge, mlp=model)
    for name, m in models.items():
        write_json(directory/(name+".json"), m)
        if not np.allclose(predict(m, vx), predict(load_model(directory/(name+".json")), vx), rtol=0, atol=1e-14):
            raise RuntimeError("model roundtrip prediction mismatch")
    selection = dict(trials=trials, ridge_candidates=[dict(alpha=m["alpha"],
        validation=metrics(vy, predict(m,vx)), training_seconds=m["training_seconds"]) for m in ridge_candidates],
        selected_widths=model["widths"], selected_alpha=model["alpha"], primary_seed=model["seed"],
        selected_seed_models=selected, reproducibility_max_difference=reproduction_difference,
        reproduction_training_seconds=repeated["training_seconds"],
        search_training_seconds=sum(t["training_seconds"] for t in trials)+sum(m["training_seconds"] for m in ridge_candidates)+dummy["training_seconds"],
        serialization_verified=True, test_used_for_selection=False)
    write_json(path, selection)
    return models, selection


def distribution(samples):
    a = np.asarray(samples, dtype=float)
    return dict(samples_seconds=a.tolist(), median_seconds=float(np.median(a)),
                q25_seconds=float(np.quantile(a,.25)), q75_seconds=float(np.quantile(a,.75)),
                min_seconds=float(a.min()), max_seconds=float(a.max()))


def time_prediction(model, x, loops=1000, repeats=7):
    predict(model, x)
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        for _ in range(loops):
            predict(model, x)
        samples.append((perf_counter()-start)/loops)
    return distribution(samples)


def benchmark(directory, models, x, config, selection, plan):
    if config.get("version") == VERSION_V2:
        return benchmark_v2(directory, models, x, config, selection, plan)
    path = directory/"timing.json"
    if path.exists():
        return read_json(path)
    timing = dict(prediction={}, mc_api={}, mc_transport={})
    for name, m in models.items():
        timing["prediction"][name] = dict(single=time_prediction(m, [[1.5]]),
                                         batch=time_prediction(m, x, loops=500), batch_size=len(x))
    layer = water()
    # Warm the exact numerical kernel before each paired timing series.
    physics.simulate_slab([layer], [0., 1.5], 100, seed=88000, material_mode="reference")
    for label, n in (("single_repeat", config["histories_per_repeat"]),
                     ("pooled_precision", config["histories_per_repeat"]*config["repeats"])):
        api, kernel = [], []
        for i in range(7):
            start = perf_counter()
            r = physics.simulate_slab([layer], [0., 1.5], n, seed=88001+i, material_mode="reference")
            api.append(perf_counter()-start)
            kernel.append(r["timing"]["transport_seconds"])
        timing["mc_api"][label] = dict(histories=n, **distribution(api))
        timing["mc_transport"][label] = distribution(kernel)
    # Include request variation across held-out thicknesses; no export or plotting.
    varied = []
    for i, width in enumerate(x[:, 0]):
        start = perf_counter()
        physics.simulate_slab([layer], [0., float(width)], config["histories_per_repeat"],
                              seed=89000+i, material_mode="reference")
        varied.append(perf_counter()-start)
    timing["mc_test_distribution"] = distribution(varied)
    loads = []
    for _ in range(7):
        start = perf_counter()
        load_model(directory/"mlp.json")
        loads.append(perf_counter()-start)
    timing["model_load"] = distribution(loads)
    generation = read_json(directory/"generation.json")
    predict_seconds = timing["prediction"]["mlp"]["single"]["median_seconds"]
    mc_seconds = timing["mc_api"]["single_repeat"]["median_seconds"]
    setup = generation["simulation_seconds"] + models["mlp"]["preprocessing_seconds"] + models["mlp"]["training_seconds"]
    research_setup = (generation["current_generation_wall_seconds"] + selection["search_training_seconds"]
        + selection["reproduction_training_seconds"] + sum(r["wrapper_seconds"] for r in plan["records"]))
    timing.update(dataset_simulation_seconds=generation["simulation_seconds"],
        dataset_generation_wall_seconds=generation["current_generation_wall_seconds"],
        selected_training_seconds=models["mlp"]["training_seconds"],
        preprocessing_seconds=models["mlp"]["preprocessing_seconds"],
        first_call=plan["first_call_including_possible_jit"],
        speedup_api=mc_seconds/predict_seconds,
        speedup_transport=timing["mc_transport"]["single_repeat"]["median_seconds"]/predict_seconds,
        setup_seconds=setup, research_setup_seconds=research_setup,
        break_even_queries=setup/(mc_seconds-predict_seconds) if mc_seconds>predict_seconds else None,
        research_break_even_queries=research_setup/(mc_seconds-predict_seconds) if mc_seconds>predict_seconds else None,
        conditional=True, caveat="Not accuracy-equivalent savings; fixed 1.5 cm workload, MC sampling error and surrogate bias differ.")
    write_json(path, timing)
    return timing


def evaluate_models(models, arrays, grouped, selection, partitions):
    result = {name: {part: metrics(y, predict(model,x)) for part,(x,y) in arrays.items()}
              for name,model in models.items()}
    x, y = arrays["test"]
    predicted = predict(models["mlp"],x)
    rng = np.random.default_rng(4815)
    boots = [metrics(y[idx],predicted[idx]) for idx in rng.integers(0,len(y),(2000,len(y)))]
    result["bootstrap"] = {key: np.quantile([r[key] for r in boots],[.025,.975]).tolist()
                           for key in ("mae","rmse","bias")}
    result["bootstrap"]["unit"] = "independent configuration; fixed trained model; conditional on noisy MC labels"
    result["seed_sensitivity"] = [dict(seed=m["seed"], test=metrics(y,predict(m,x)))
                                  for m in selection["selected_seed_models"]]
    by_group = {r["configuration_id"]:r for r in grouped}
    test_rows = [by_group[g] for g in partitions["test"]]
    se = np.array([r["standard_error"] for r in test_rows])
    residual = predicted-y
    result["noise"] = dict(pooled_se_median=float(np.median(se)),
        pooled_noise_rms=float(np.sqrt(np.mean(se**2))),
        replicate_sem_median=float(np.median([r["replicate_sem"] for r in test_rows])),
        within_mc95_fraction=float(np.mean([(r["ci_low"]<=p<=r["ci_high"]) for r,p in zip(test_rows,predicted)])),
        note="MC intervals are not calibrated surrogate predictive intervals; three repeats give imprecise empirical SEM")
    bins = []
    for low, high in zip(np.linspace(*DOMAIN,4)[:-1],np.linspace(*DOMAIN,4)[1:]):
        mask = (x[:,0]>=low)&(x[:,0]<=high)
        bins.append(dict(lower_cm=float(low),upper_cm=float(high),n=int(mask.sum()),
                         metrics=metrics(y[mask],predicted[mask]) if mask.any() else None))
    result["error_by_thickness"] = bins
    result["educational_accuracy_pass"] = result["mlp"]["test"]["rmse"]<=.01
    result["test_predictions"] = [dict(**r,prediction=float(p),residual=float(e)) for r,p,e in zip(test_rows,predicted,residual)]
    return result


def plots(directory, model, results, timing):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.facecolor":"white","axes.grid":True,"grid.alpha":.2})

    def finish(fig, name):
        fig.savefig(directory/(name+".png"),dpi=220,bbox_inches="tight")
        plt.close(fig)

    history=model["history"]
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    for key,label in (("train_rmse","Training RMSE"),("validation_rmse","Validation RMSE")):
        ax.plot([r["iteration"] for r in history],[r[key] for r in history],label=label)
    ax.set(xlabel="L-BFGS iteration",ylabel="RMSE (transmission fraction)",yscale="log",title="Optimization history; test not monitored")
    ax.legend(); finish(fig,"learning_curve")
    rows=results["test_predictions"]
    y=np.array([r["target"] for r in rows]); p=np.array([r["prediction"] for r in rows])
    x=np.array([r["thickness_cm"] for r in rows])
    fig,ax=plt.subplots(figsize=(5.5,5),layout="constrained")
    ax.errorbar(y,p,xerr=1.96*np.array([r["standard_error"] for r in rows]),fmt="o",alpha=.75,label="Test; horizontal approx. 95% MC error")
    ax.plot([0,1],[0,1],"--",color="black",label="Identity")
    ax.set(xlabel="Pooled Monte Carlo transmission",ylabel="MLP transmission",title="Independent configurations")
    ax.legend(fontsize=9); finish(fig,"predicted_actual")
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    ax.errorbar(x,p-y,yerr=1.96*np.array([r["standard_error"] for r in rows]),fmt="o",label="Residual ± approx. MC error")
    ax.axhline(0,color="black",ls="--"); ax.set(xlabel="Thickness (cm)",ylabel="MLP − MC fraction",title="Test residuals; no clipping")
    ax.legend(); finish(fig,"residuals")
    names=["dummy","ridge","mlp"]
    fig,ax=plt.subplots(figsize=(6,4),layout="constrained")
    ax.bar(names,[results[n]["test"]["rmse"] for n in names],color=["#888888","#b27b40","#2266aa"])
    ax.set(ylabel="Test RMSE (fraction)",yscale="log",title="Baselines and MLP; logarithmic scale")
    finish(fig,"model_comparison")
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    measures=[timing["mc_api"]["single_repeat"],timing["prediction"]["mlp"]["single"],timing["prediction"]["mlp"]["batch"]]
    divisors=[1,1,timing["prediction"]["mlp"]["batch_size"]]
    values=[m["median_seconds"]*1e6/d for m,d in zip(measures,divisors)]
    errors=np.array([[max(0,m["median_seconds"]-m["q25_seconds"])*1e6/d for m,d in zip(measures,divisors)],
                     [max(0,m["q75_seconds"]-m["median_seconds"])*1e6/d for m,d in zip(measures,divisors)]])
    ax.bar(["Warm MC API","MLP single","MLP batch / item"],values,yerr=errors,capsize=5)
    ax.set(ylabel="Microseconds / configuration (log)",yscale="log",title="Median and IQR; different approximation errors")
    finish(fig,"timing_comparison")
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    limit=timing["research_break_even_queries"] or 10000
    queries=np.linspace(0,limit*1.5,200)
    ax.plot(queries,queries*timing["mc_api"]["single_repeat"]["median_seconds"],label="Repeated warm MC")
    prediction=timing["prediction"]["mlp"]["single"]["median_seconds"]
    ax.plot(queries,timing["setup_seconds"]+queries*prediction,label="ML: simulation + selected fit")
    ax.plot(queries,timing["research_setup_seconds"]+queries*prediction,label="ML: pilot, export, search, repeated fit")
    ax.set(xlabel="Number of queries",ylabel="Accumulated compute time (s)",title="Conditional break-even; accuracy equivalence not established")
    ax.legend(fontsize=9); finish(fig,"break_even")


def run(mode, directory):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"metrics.json").exists():
        raise FileExistsError("completed results preserved; use evaluate/predict or a new output directory")
    plan=pilot(directory,mode=="quick")
    config, records, grouped=generate(directory,plan)
    arrays=partition_arrays(grouped,config["partitions"])
    models,selection=train(directory,arrays,config)
    # Model and selection are saved before opening the test targets for evaluation.
    results=evaluate_models(models,arrays,grouped,selection,config["partitions"])
    timing=benchmark(directory,models,arrays["test"][0],config,selection,plan)
    results["checks"]=dict(source_hashes_unchanged=hashes()==config["source_hashes"],
        group_disjoint=True,export_reload_all_runs=True,model_roundtrip=True,
        scaler_train_only=bool(np.array_equal(np.asarray(models["mlp"]["mean"]),arrays["train"][0].mean(axis=0))),
        repeated_training_max_difference=selection["reproducibility_max_difference"],
        reference_flags_preserved=all(not r["certified_training_eligible"] for r in records))
    write_json(directory/"metrics.json",results)
    save_csv(directory/"comparison.csv",[dict(model=n,**results[n]["test"],
        training_seconds=models[n]["training_seconds"],
        single_predict_seconds=timing["prediction"][n]["single"]["median_seconds"]) for n in models])
    save_csv(directory/"test_predictions.csv",results["test_predictions"])
    plots(directory,models["mlp"],results,timing)
    print(json.dumps({"output":str(directory),"mlp_test":results["mlp"]["test"],
        "speedup":timing["speedup_api"],"conditional_break_even":timing["break_even_queries"]},indent=2))


def atomic_json(path, value):
    """Replace only a known checkpoint owned by this run, never a raw bundle."""
    import os
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".ml-checkpoint-", suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def design_points(n, seed=V2_DESIGN_SEED):
    bounds = np.asarray([s["domain"] for s in feature_spec()])
    points = qmc.scale(qmc.LatinHypercube(3, rng=np.random.default_rng(seed)).random(n),
                       bounds[:, 0], bounds[:, 1])
    return inputs(points, feature_spec=feature_spec())


def pilot_v2(directory, quick, budget_seconds):
    path = directory/"pilot.json"
    if path.exists():
        plan = read_json(path)
        if plan["source_hashes"] != hashes() or plan["quick"] != quick or plan["budget_seconds"] != budget_seconds:
            raise ValueError("pilot resume requires identical source, mode and budget")
        return plan
    started = perf_counter()
    # Paired one-factor checks from independent histories; not a causal ML claim.
    center = [1.55, .61, 1.]
    points = [center]
    for j, spec in enumerate(feature_spec()):
        for bound in spec["domain"]:
            point = center.copy()
            point[j] = bound
            points.append(point)
    points += design_points(8, 71119).tolist()
    records, calls = [], []
    for i, point in enumerate(points):
        for repeat in range(3):
            c = configuration(None, 720000+i*3+repeat, 2000, feature_values=point)
            tick = perf_counter()
            experiment = physics.run_slab_experiment(c)
            calls.append(perf_counter()-tick)
            records.append(observation(experiment))
    pooled = aggregate(records)
    cost = float(np.median(calls[1:]))
    # Budget covers main generation invocation, not optimization/search or tests.
    # The conservative factor includes construction + native export/reload.
    histories = 1000 if quick else 8000
    cap, floor = (60, 60) if quick else (900, 300)
    estimate = budget_seconds / (3 * (cost * histories/2000 + .045))
    groups = int(min(cap, max(floor, 60 * int(estimate/60))))
    if estimate < floor:
        raise ValueError("pilot budget insufficient for minimum design; increase --budget-seconds")
    response = {}
    lookup = {tuple(r[n] for n in [s["name"] for s in feature_spec()]): r for r in pooled}
    for j, spec in enumerate(feature_spec()):
        low, high = center.copy(), center.copy()
        low[j], high[j] = spec["domain"]
        left, right = lookup[tuple(low)], lookup[tuple(high)]
        delta = right["target"]-left["target"]
        uncertainty = np.hypot(left["standard_error"], right["standard_error"])
        response[spec["name"]] = dict(delta=float(delta), combined_se=float(uncertainty),
                                       resolved=bool(abs(delta)>3*uncertainty))
    if not all(r["resolved"] for r in response.values()):
        raise RuntimeError("pilot did not resolve all feature effects; inspect before main generation")
    plan = dict(version=VERSION_V2, feature_spec=feature_spec(), source_hashes=hashes(),
        quick=quick, budget_seconds=budget_seconds, groups=groups, repeats=3,
        histories_per_repeat=histories, generator_version=VERSION_V2,
        wall_seconds=perf_counter()-started, warm_wrapper=distribution(calls[1:]),
        cold_first_wrapper_seconds=calls[0], cold_first_api_seconds=records[0]["seconds"],
        cold_first_transport_seconds=records[0]["transport_seconds"],
        first_call_scope="first call in fresh process; includes JIT and initialization, not isolated compiler time",
        records=records, response=response, target_range=[min(r["target"] for r in pooled),max(r["target"] for r in pooled)],
        typical_se=float(np.median([r["standard_error"] for r in pooled])),
        design="scrambled Latin hypercube; comparison: independent random leaves marginal gaps, 3-D grid grows cubically",
        bounds_reason="positive absorption avoids nearly conservative long histories; max total optical width 9.6; fixed ranges before test")
    write_json(path, plan)
    print(f"V2 pilot: {groups} groups x 3 x {histories} histories; all three feature effects resolved", flush=True)
    return plan


def generate_v2(directory, plan):
    path = directory/"experiment_config.json"
    if path.exists():
        config = read_json(path)
        if config["source_hashes"] != hashes() or config["feature_spec"] != feature_spec():
            raise ValueError("generation resume requires identical source and feature schema")
    else:
        points = design_points(plan["groups"])
        groups = [configuration(None, 800000+i*3, plan["histories_per_repeat"],
                                feature_values=point)["group_id"] for i, point in enumerate(points)]
        if len(set(groups)) != len(points):
            raise ValueError("duplicate physical configuration")
        partitions = split_groups(groups, seed=V2_DESIGN_SEED)
        config = dict(version=VERSION_V2, dataset_schema=VERSION_V2, parameter_schema="explicit_rates_v2",
            source_hashes=hashes(), feature_spec=feature_spec(), feature_names=[s["name"] for s in feature_spec()],
            points=points.tolist(), groups=groups, partitions=partitions,
            repeats=3, histories_per_repeat=plan["histories_per_repeat"], seed_base=800000,
            design_seed=V2_DESIGN_SEED, split_seed=V2_DESIGN_SEED, training_seeds=list(SEEDS),
            candidates=[dict(hidden=list(h), alpha=a) for h,a in CANDIDATES_V2],
            maxiter=2500, ftol=1e-10, gtol=1e-6, ridge_alphas=[0., .01, 1.],
            selection_rule="minimum mean validation RMSE over fixed seeds; only fully converged candidates; primary seed 17",
            studies=dict(train_fractions=[.2,.4,.6,.8,1.],subset_seed=1942,
                ablation=[[0],[0,1],[0,2],[0,1,2]],bootstrap_repeats=2000,bootstrap_seed=4815,
                diagnostic_lhs=24 if not plan["quick"] else 8, diagnostic_slice_side=7 if not plan["quick"] else 3,
                diagnostic_histories=100000 if not plan["quick"] else 5000,
                noise_histories=[500,2000,8000] if not plan["quick"] else [250,500,1000],noise_repeats=8),
            hypotheses=dict(H1="MLP improves RMSE versus linear; paired bootstrap difference CI excludes zero; polynomial comparison also reported",
                H2="mean validation RMSE at 100% below 20%; no strict monotonicity assumed",
                H3="removing input(s) worsens mean validation RMSE; predictive, not causal importance",
                H4="positive MC-predict timing difference; conditional setup cost amortization",
                H5="compare label SE, repeated-history scaling and independent high-precision diagnostic errors"),
            acceptable_test_rmse=.015, acceptable_max_error=.06,
            scientific_validation="not_established", source_mu=1., source_x=0., max_collisions=1000,
            optional_source_mu="not varied: three-parameter study completed first; public source_mu API remains unchanged",
            backend="numba" if physics.HAS_NUMBA else "python",
            quick=plan["quick"], budget_seconds=plan["budget_seconds"],
            environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                numba=getattr(sys.modules.get("numba"),"__version__",None),
                platform=platform.platform(), processor=platform.processor(),
                omp_threads=__import__("os").environ.get("OMP_NUM_THREADS"),
                openblas_threads=__import__("os").environ.get("OPENBLAS_NUM_THREADS")),
            reproduction=f"python -B -m benchmarks.ml_baseline {'quick' if plan['quick'] else 'full'} --schema v2 --budget-seconds {plan['budget_seconds']} --output <new-directory>")
        write_json(path, config)
        write_json(directory/"splits.json", partitions)
    progress_path = directory/"generation_progress.json"
    progress = read_json(progress_path) if progress_path.exists() else dict(wall_seconds=0., sessions=0)
    started, previous = perf_counter(), progress["wall_seconds"]
    rows = []
    existing = set()
    archive_path = directory/"experiments.zip"
    with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        existing.update(archive.namelist())
        for i, point in enumerate(config["points"]):
            for repeat in range(config["repeats"]):
                prefix = f"g{i:04d}_r{repeat}"
                members = {prefix+"/"+n for n in ("experiment.json","results.csv","arrays.npz")}
                if members <= existing:
                    continue
                if members & existing:
                    raise ValueError("partial bundle preserved; use a new output directory")
                if perf_counter()-started > config["budget_seconds"]:
                    atomic_json(progress_path,dict(wall_seconds=previous+perf_counter()-started,sessions=progress["sessions"]+1))
                    raise TimeoutError("generation invocation budget exhausted; complete bundles saved; rerun same command to resume")
                c = configuration(None,config["seed_base"]+i*3+repeat,config["histories_per_repeat"],feature_values=point)
                try:
                    experiment = physics.run_slab_experiment(c)
                    row = observation(experiment)
                    with tempfile.TemporaryDirectory(prefix="heliosmc-v2-export-") as tmp:
                        bundle=Path(tmp)/"bundle"
                        physics.export_slab_experiment(experiment,bundle)
                        if physics.load_slab_experiment(bundle) != experiment:
                            raise RuntimeError("native export roundtrip mismatch")
                        for item in sorted(bundle.iterdir()):
                            archive.write(item,prefix+"/"+item.name)
                    rows.append(row)
                    existing.update(members)
                except Exception as exc:
                    atomic_json(directory/"generation_failure.json",dict(configuration=c,reason=str(exc)))
                    raise
            if (i+1)%60==0:
                print(f"V2 archived {i+1}/{len(config['points'])} groups",flush=True)
    wall=previous+perf_counter()-started
    atomic_json(progress_path,dict(wall_seconds=wall,sessions=progress["sessions"]+1))
    # Reload every archived run, including resumed bundles, through native checks.
    tick=perf_counter()
    records, grouped=load_dataset(directory)
    aggregation_seconds=perf_counter()-tick
    if set(r["configuration_id"] for r in grouped)!=set(config["groups"]) or any(r["repeats"]!=3 for r in grouped):
        raise ValueError("incomplete/mismatched dataset")
    if not (directory/"generation.json").exists():
        write_json(directory/"generation.json",dict(wall_seconds=wall,aggregation_seconds=aggregation_seconds,
            sessions=progress["sessions"]+1,simulation_seconds=sum(r["seconds"] for r in records),
            transport_seconds=sum(r["transport_seconds"] for r in records),
            excluded_records=[],groups=len(grouped),runs=len(records)))
        save_csv(directory/"dataset.csv",grouped)
        save_csv(directory/"mc_records.csv",[{k:v for k,v in r.items() if k not in ("warnings","code_sha256")} for r in records])
    return config,records,grouped


def model_summary(model, x, y, vx, vy):
    return dict(kind=model["kind"],widths=model.get("widths"),alpha=model["alpha"],seed=model["seed"],
        success=model["success"],iterations=model["iterations"],message=model.get("message","closed-form least squares"),
        training_seconds=model["training_seconds"],preprocessing_seconds=model["preprocessing_seconds"],
        train=metrics(y,predict(model,x)),validation=metrics(vy,predict(model,vx)),
        gradient_max=model.get("gradient_max"))


def train_v2(directory, arrays, config):
    stage_started=perf_counter()
    path=directory/"selection.json"
    if path.exists():
        return {n:load_model(directory/(n+".json")) for n in ("dummy","ridge","polynomial_ridge","mlp")}, read_json(path)
    x,y=arrays["train"]; vx,vy=arrays["validation"]
    spec=feature_spec()
    models={"dummy":fit_model(x,y,"dummy",feature_spec=spec)}
    trials=[]; candidates=[]; ridge_trials=[]
    for kind in ("ridge","polynomial_ridge"):
        fitted=[fit_model(x,y,kind,alpha=a,feature_spec=spec) for a in config["ridge_alphas"]]
        models[kind]=min(fitted,key=lambda m:metrics(vy,predict(m,vx))["rmse"])
        ridge_trials.extend(model_summary(m,x,y,vx,vy) for m in fitted)
    checkpoint=directory/"training_trials.json"
    done=read_json(checkpoint) if checkpoint.exists() else []
    for hidden,alpha in CANDIDATES_V2:
        fitted=[]
        for seed in SEEDS:
            matches=[m for m in done if m.get("widths")==[3,*hidden,1] and m["alpha"]==alpha and m["seed"]==seed]
            if matches:
                m=matches[0]
            else:
                m=fit_model(x,y,hidden=hidden,alpha=alpha,seed=seed,validation=(vx,vy),
                            maxiter=config["maxiter"],feature_spec=spec)
                done.append(m)
                atomic_json(checkpoint,done)
            fitted.append(m)
            trials.append(model_summary(m,x,y,vx,vy))
        score=float(np.mean([metrics(vy,predict(m,vx))["rmse"] for m in fitted]))
        candidates.append((score,fitted))
        print(f"V2 MLP {hidden}, alpha={alpha}: val={score:.6f}, converged={[m['success'] for m in fitted]}",flush=True)
    valid=[c for c in candidates if all(m["success"] for m in c[1])]
    if not valid:
        atomic_json(directory/"training_failure.json",dict(trials=trials,reason="no fully converged candidate; no test evaluation performed"))
        raise RuntimeError("no converged candidate; preserve diagnostics")
    score,selected=min(valid,key=lambda c:c[0]); model=selected[0]
    repeated=fit_model(x,y,hidden=tuple(model["widths"][1:-1]),alpha=model["alpha"],seed=17,
                       validation=(vx,vy),maxiter=config["maxiter"],feature_spec=spec)
    difference=float(np.max(abs(predict(model,vx)-predict(repeated,vx))))
    if not repeated["success"] or difference>1e-10:
        raise RuntimeError("repeated training mismatch")
    model["experiment"]=dict(version=VERSION_V2,source_hashes=config["source_hashes"],
        train_groups=config["partitions"]["train"],validation_groups=config["partitions"]["validation"],
        environment=config["environment"],scientific_validation="not_established")
    models["mlp"]=model
    for name,m in models.items():
        if not (directory/(name+".json")).exists():
            write_json(directory/(name+".json"),m)
        loaded=load_model(directory/(name+".json"))
        if not np.array_equal(predict(m,vx),predict(loaded,vx)):
            raise RuntimeError("model serialization mismatch")
    selection=dict(trials=trials,ridge_trials=ridge_trials,selected_seed_models=selected,
        selected_widths=model["widths"],selected_alpha=model["alpha"],mean_validation_rmse=score,
        primary_seed=17,test_used_for_selection=False,reproducibility_max_difference=difference,
        reproduction_training_seconds=repeated["training_seconds"],
        search_training_seconds=sum(m["training_seconds"]+m["preprocessing_seconds"] for m in done)
            +sum(m["training_seconds"]+m["preprocessing_seconds"] for m in ridge_trials)
            +models["dummy"]["training_seconds"],
        stage_wall_seconds=perf_counter()-stage_started,
        stage_wall_scope="this invocation incl checkpoints and roundtrip, excluding final selection.json write")
    write_json(path,selection)
    save_csv(directory/"learning_curves.csv",model["history"])
    return models,selection


def studies_v2(directory, arrays, model, selection, config):
    """All study fits use only train and validation; test metrics added later."""
    started=perf_counter()
    path=directory/"study_models.json"
    studies=read_json(path) if path.exists() else []
    x,y=arrays["train"]; vx,vy=arrays["validation"]
    permutation=np.random.default_rng(config["studies"]["subset_seed"]).permutation(len(x))
    tasks=[("size",fraction,[0,1,2]) for fraction in config["studies"]["train_fractions"]]
    tasks += [("ablation",1.,cols) for cols in config["studies"]["ablation"]]
    for study,fraction,cols in tasks:
        count=max(10,round(len(x)*fraction)); idx=permutation[:count]
        for seed in SEEDS:
            key=f"{study}:{fraction}:{cols}:{seed}"
            if any(r["key"]==key for r in studies):
                continue
            if fraction==1. and cols==[0,1,2]:
                fitted=next(m for m in selection["selected_seed_models"] if m["seed"]==seed)
                indices=np.arange(len(x))
            else:
                indices=idx
                fitted=fit_model(x[idx][:,cols],y[idx],hidden=tuple(model["widths"][1:-1]),
                    alpha=model["alpha"],seed=seed,validation=(vx[:,cols],vy),
                    maxiter=config["maxiter"],feature_spec=feature_spec([config["feature_names"][j] for j in cols]))
            studies.append(dict(key=key,study=study,fraction=fraction,columns=cols,n_groups=len(indices),
                training_histories=len(indices)*config["repeats"]*config["histories_per_repeat"],
                group_ids=[config["partitions"]["train"][i] for i in indices],model=fitted,
                validation=metrics(vy,predict(fitted,vx[:,cols])),success=fitted["success"]))
            atomic_json(path,studies)
        print(f"V2 {study}: fraction={fraction}, columns={cols}, completed 3 seeds",flush=True)
    execution=directory/"study_execution.json"
    previous=read_json(execution) if execution.exists() else dict(wall_seconds=0.,sessions=0)
    atomic_json(execution,dict(wall_seconds=previous["wall_seconds"]+perf_counter()-started,
        sessions=previous["sessions"]+1,scope="completed study invocations incl model checkpoints"))
    return studies


def bootstrap_scores(y, predictions, seed=4815, repetitions=2000):
    indices=np.random.default_rng(seed).integers(0,len(y),(repetitions,len(y)))
    results={}
    rmse_samples={}
    for name,prediction in predictions.items():
        residual=prediction[indices]-y[indices]
        values=dict(mae=np.mean(abs(residual),axis=1),rmse=np.sqrt(np.mean(residual**2,axis=1)),
                    bias=np.mean(residual,axis=1))
        results[name]={k:np.quantile(v,[.025,.975]).tolist() for k,v in values.items()}
        rmse_samples[name]=values["rmse"]
    for baseline in ("ridge","polynomial_ridge"):
        if baseline in rmse_samples:
            results["mlp_minus_"+baseline]=np.quantile(rmse_samples["mlp"]-rmse_samples[baseline],[.025,.975]).tolist()
    return dict(intervals=results,repetitions=repetitions,seed=seed,
                unit="independent configuration; percentile; fixed trained models and noisy MC labels; paired differences")


def evaluate_v2(directory, models, arrays, grouped, selection, studies, config):
    result={name:{part:metrics(y,predict(model,x)) for part,(x,y) in arrays.items()} for name,model in models.items()}
    x,y=arrays["test"]; predictions={n:predict(m,x) for n,m in models.items()}
    result["bootstrap"]=bootstrap_scores(y,predictions)
    result["seed_sensitivity"]=[dict(seed=m["seed"],success=m["success"],
        **metrics(y,predict(m,x))) for m in selection["selected_seed_models"]]
    size_rows=[]; ablation_rows=[]
    reference={r["seed"]:r["rmse"] for r in result["seed_sensitivity"]}
    for study in studies:
        fitted=study["model"]
        test=metrics(y,predict(fitted,x[:,study["columns"]]))
        row=dict(fraction=study["fraction"],features=" + ".join(fitted["features"]),n_features=len(fitted["features"]),
            n_groups=study["n_groups"],training_histories=study["training_histories"],seed=fitted["seed"],
            success=fitted["success"],iterations=fitted["iterations"],training_seconds=fitted["training_seconds"],
            validation_rmse=study["validation"]["rmse"],test_rmse=test["rmse"],
            rmse_change_from_full=test["rmse"]-reference[fitted["seed"]])
        (size_rows if study["study"]=="size" else ablation_rows).append(row)
    save_csv(directory/"dataset_size_study.csv",size_rows)
    save_csv(directory/"ablation.csv",ablation_rows)
    result["dataset_size_study"]=size_rows; result["ablation"]=ablation_rows
    by_group={r["configuration_id"]:r for r in grouped}
    test_rows=[by_group[g] for g in config["partitions"]["test"]]
    residual=predictions["mlp"]-y
    result["test_predictions"]=[dict(**r,**{n+"_prediction":float(predictions[n][i]) for n in predictions},
                                    residual=float(residual[i])) for i,r in enumerate(test_rows)]
    error_rows=[]
    for j,spec in enumerate(feature_spec()):
        for b,(lo,hi) in enumerate(zip(np.linspace(*spec["domain"],4)[:-1],np.linspace(*spec["domain"],4)[1:])):
            mask=(x[:,j]>=lo)&((x[:,j]<=hi) if b==2 else (x[:,j]<hi))
            if mask.any():
                error_rows.append(dict(feature=spec["name"],lower=float(lo),upper=float(hi),n=int(mask.sum()),
                                      **metrics(y[mask],predictions["mlp"][mask])))
    result["error_regions"]=error_rows
    save_csv(directory/"error_regions.csv",error_rows)
    se=np.array([r["standard_error"] for r in test_rows])
    result["label_noise"]=dict(pooled_se_median=float(np.median(se)),pooled_se_rms=float(np.sqrt(np.mean(se**2))),
        replicate_sem_median=float(np.median([r["replicate_sem"] for r in test_rows])),
        mc95_coverage=float(np.mean([r["ci_low"]<=p<=r["ci_high"] for r,p in zip(test_rows,predictions["mlp"])])),
        caveat="MC intervals describe labels; not calibrated MLP predictive uncertainty")
    result["accuracy_pass"]=result["mlp"]["test"]["rmse"]<=config["acceptable_test_rmse"] and result["mlp"]["test"]["max_absolute_error"]<=config["acceptable_max_error"]
    return result


def diagnostic_v2(directory, models, config):
    path=directory/"diagnostics.json"
    if path.exists():
        return read_json(path)
    started=perf_counter(); protocol=config["studies"]
    points=design_points(protocol["diagnostic_lhs"],91419).tolist()
    labels=["independent_lhs"]*len(points)
    # Observed 2-D slice, not an interpolated error heatmap: s=1 cm^-1 fixed.
    for width in np.linspace(.1,3.,protocol["diagnostic_slice_side"]):
        for absorption in np.linspace(.02,1.2,protocol["diagnostic_slice_side"]):
            points.append([float(width),float(absorption),1.]); labels.append("slice_s1")
    rows=[]; noise=[]
    with zipfile.ZipFile(directory/"diagnostic_experiments.zip","a",compression=zipfile.ZIP_DEFLATED) as archive:
        existing=set(archive.namelist())

        def run_bundle(prefix,c):
            members={prefix+"/"+n for n in ("experiment.json","results.csv","arrays.npz")}
            with tempfile.TemporaryDirectory(prefix="heliosmc-v2-diagnostic-") as tmp:
                bundle=Path(tmp)/"bundle"
                if members<=existing:
                    return read_bundle(archive,prefix,bundle)
                if members & existing:
                    raise ValueError("partial diagnostic bundle")
                experiment=physics.run_slab_experiment(c)
                physics.export_slab_experiment(experiment,bundle)
                if physics.load_slab_experiment(bundle)!=experiment:
                    raise RuntimeError("diagnostic export roundtrip mismatch")
                for item in sorted(bundle.iterdir()):
                    archive.write(item,prefix+"/"+item.name)
                existing.update(members)
                return experiment

        for i,(point,label) in enumerate(zip(points,labels)):
            c=configuration(None,1200000+i,protocol["diagnostic_histories"],feature_values=point)
            if c["group_id"] in config["groups"]:
                raise ValueError("diagnostic overlap with main dataset")
            experiment=run_bundle(f"diagnostic_{i:04d}",c)
            row=observation(experiment)
            row.update(diagnostic=label,**{name+"_prediction":float(predict(m,[point])[0]) for name,m in models.items()})
            rows.append(row)
        noise_points=design_points(8 if not config["quick"] else 3,91519)
        for i,point in enumerate(noise_points):
            for j,histories in enumerate(protocol["noise_histories"]):
                runs=[]
                for repeat in range(protocol["noise_repeats"]):
                    c=configuration(None,1400000+i*100+j*10+repeat,histories,feature_values=point)
                    runs.append(observation(run_bundle(f"noise_{i}_{j}_{repeat}",c)))
                values=np.array([r["target"] for r in runs]); pooled=aggregate(runs)[0]
                noise.append(dict(point=i,histories=histories,repeats=len(runs),target=float(values.mean()),
                    replicate_sd=float(values.std(ddof=1)),mean_plugin_se=float(np.mean([r["standard_error"] for r in runs])),
                    pooled_se=pooled["standard_error"],group_id=c["group_id"]))
        # Known analytic limit is checked separately, outside the v2 model domain
        # when absorption=0 or width=0; only s=0 is within this domain.
        analytic=[]
        for i,(width,a) in enumerate(((.1,.02),(1.,.5),(3.,1.2))):
            c=physics.create_slab_configuration("v2-pure-absorption-check",[synthetic_medium(a,0.)],
                [0.,width],20000,seed=1600000+i,verification="pure_absorption_v1")
            experiment=run_bundle(f"analytic_{i}",c)
            analytic.append(dict(width_cm=width,absorption_rate=a,expected=float(np.exp(-a*width)),
                estimate=experiment["result"]["estimates"]["escaped_right"]["estimate"],
                **experiment["verification"]))
    scores={label:{name:metrics([r["target"] for r in rows if r["diagnostic"]==label],
                               [r[name+"_prediction"] for r in rows if r["diagnostic"]==label]) for name in models}
            for label in ("independent_lhs","slice_s1")}
    result=dict(rows=rows,noise_study=noise,scores=scores,analytic_checks=analytic,
        noise_aggregate=[dict(histories=n,empirical_rms_sd=float(np.sqrt(np.mean([r["replicate_sd"]**2 for r in noise if r["histories"]==n]))),
            mean_plugin_se=float(np.mean([r["mean_plugin_se"] for r in noise if r["histories"]==n]))) for n in protocol["noise_histories"]],
        diagnostic_se_median=float(np.median([r["standard_error"] for r in rows if r["diagnostic"]=="independent_lhs"])),
        wall_seconds=perf_counter()-started,scope="frozen models; new in-domain configurations, separate observed s=1 slice; no tuning")
    write_json(path,result)
    save_csv(directory/"noise_study.csv",noise)
    save_csv(directory/"diagnostic_predictions.csv",[{k:v for k,v in r.items() if k not in ("warnings","code_sha256")} for r in rows])
    return result


def benchmark_v2(directory, models, x, config, selection, plan):
    path=directory/"timing.json"
    if path.exists():
        return read_json(path)
    records=[]; tick=perf_counter()
    specs=feature_spec()
    layers=[synthetic_medium(point[1],point[2]) for point in x]
    preparation=perf_counter()-tick
    physics.simulate_slab([layers[0]],[0.,float(x[0,0])],100,seed=1700000)
    # Seven matched sweeps; timing matrices explicitly retain the same inputs.
    for repeat in range(7):
        api=[]; kernel=[]; prediction={name:[] for name in models}
        for i,point in enumerate(x):
            tick=perf_counter()
            result=physics.simulate_slab([layers[i]],[0.,float(point[0])],config["histories_per_repeat"],seed=1700001+repeat*len(x)+i)
            api.append(perf_counter()-tick); kernel.append(result["timing"]["transport_seconds"])
        for name,model in models.items():
            tick=perf_counter()
            for _ in range(10):
                for point in x:
                    predict(model,point[None,:])
            prediction[name]=(perf_counter()-tick)/(10*len(x))
        records.append(dict(repeat=repeat,api=api,kernel=kernel,predict_seconds=prediction))
    result=dict(points=x.tolist(),histories=config["histories_per_repeat"],matched_repeats=records,
        mc_api=distribution([np.mean(r["api"]) for r in records]),
        mc_transport=distribution([np.mean(r["kernel"]) for r in records]),
        request_heterogeneity=distribution([v for r in records for v in r["api"]]),
        models={},feature_preparation_seconds=preparation,
        scope="same held-out configurations; warm API incl statistics, prebuilt medium; no export, plotting or GUI")
    for name,model in models.items():
        batch=time_prediction(model,x,loops=100,repeats=7)
        loads=[]
        for _ in range(7):
            tick=perf_counter(); load_model(directory/(name+".json")); loads.append(perf_counter()-tick)
        result["models"][name]=dict(single=distribution([r["predict_seconds"][name] for r in records]),
            batch=batch,batch_size=len(x),load=distribution(loads),training_seconds=model["training_seconds"],
            preprocessing_seconds=model["preprocessing_seconds"])
    generation=read_json(directory/"generation.json")
    delta=result["mc_api"]["median_seconds"]-result["models"]["mlp"]["single"]["median_seconds"]
    deploy=generation["simulation_seconds"]+models["mlp"]["training_seconds"]+models["mlp"]["preprocessing_seconds"]+result["models"]["mlp"]["load"]["median_seconds"]
    study_models=read_json(directory/"study_models.json")
    additional=sum(r["model"]["training_seconds"]+r["model"]["preprocessing_seconds"] for r in study_models
                   if not (r["fraction"]==1. and r["columns"]==[0,1,2]))
    diagnostic=read_json(directory/"diagnostics.json")
    study_execution=read_json(directory/"study_execution.json") if (directory/"study_execution.json").exists() else dict(wall_seconds=additional)
    research=generation["wall_seconds"]+generation["aggregation_seconds"]+plan["wall_seconds"]+selection.get("stage_wall_seconds",selection["search_training_seconds"]+selection["reproduction_training_seconds"])+study_execution["wall_seconds"]+diagnostic["wall_seconds"]
    result.update(speedup=result["mc_api"]["median_seconds"]/result["models"]["mlp"]["single"]["median_seconds"],
        transport_speedup=result["mc_transport"]["median_seconds"]/result["models"]["mlp"]["single"]["median_seconds"],
        cold_call=plan["cold_first_wrapper_seconds"],generation=generation,
        search_training_seconds=selection["search_training_seconds"],study_training_seconds=additional,
        deployment_setup_seconds=deploy,research_setup_seconds=research,
        deployment_break_even=deploy/delta if delta>0 else None,research_break_even=research/delta if delta>0 else None,
        conditional=True,accuracy_equivalence_established=False,
        research_exclusions="final selection JSON serialization, timing sweeps, plotting, report generation, tests, human work; resumed interrupted fit/session wall may be incomplete")
    write_json(path,result)
    return result


def plots_v2(directory, model, results, timing, diagnostics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination=directory/"figures"; destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
        "figure.facecolor":"white","axes.grid":True,"grid.alpha":.2})

    def finish(fig,name):
        fig.savefig(destination/(name+".png"),dpi=300,bbox_inches="tight")
        plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    for key,label in (("train_rmse","Training RMSE"),("validation_rmse","Validation RMSE")):
        ax.plot([r["iteration"] for r in model["history"]],[r[key] for r in model["history"]],label=label)
    ax.set(xlabel="L-BFGS iteration",ylabel="RMSE (fraction)",yscale="log",title="Optimization; no test monitoring")
    ax.legend(); finish(fig,"learning_curve")
    rows=results["test_predictions"]; y=np.array([r["target"] for r in rows]); p=np.array([r["mlp_prediction"] for r in rows])
    fig,ax=plt.subplots(figsize=(5.6,5.3),layout="constrained")
    ax.scatter(y,p,s=18,alpha=.65); ax.plot([0,1],[0,1],"--",color="black")
    ax.set(xlabel="Pooled Monte Carlo transmission",ylabel="MLP transmission",title=f"Held-out configurations (n={len(y)})")
    finish(fig,"predicted_actual")
    fig,ax=plt.subplots(figsize=(6.5,4),layout="constrained")
    ax.hist(p-y,bins="auto",color="#3675a9",edgecolor="white")
    ax.axvline(0,color="black",ls="--"); ax.set(xlabel="MLP − MC (fraction)",ylabel="Configurations",title="Test residual distribution")
    finish(fig,"residual_distribution")
    for spec in feature_spec():
        fig,ax=plt.subplots(figsize=(6.5,4),layout="constrained")
        ax.errorbar([r[spec["name"]] for r in rows],p-y,
            yerr=1.96*np.array([r["standard_error"] for r in rows]),fmt="o",ms=3,alpha=.65,
            label="Residual ±1.96 MC SE (label uncertainty)")
        ax.axhline(0,color="black",ls="--"); ax.set(xlabel=f"{spec['name']} ({spec['unit']})",ylabel="MLP − MC (fraction)")
        ax.legend(fontsize=9); finish(fig,"residual_"+spec["name"])
    names=["dummy","ridge","polynomial_ridge","mlp"]
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    values=[results[n]["test"]["rmse"] for n in names]
    intervals=results["bootstrap"]["intervals"]
    errors=np.array([[max(0,v-intervals[n]["rmse"][0]) for n,v in zip(names,values)],
                     [max(0,intervals[n]["rmse"][1]-v) for n,v in zip(names,values)]])
    ax.bar(["Dummy","Linear Ridge","Polynomial (2)","MLP"],values,yerr=errors,capsize=4)
    ax.set(ylabel="Test RMSE (fraction)",title="Configuration bootstrap 95% intervals")
    finish(fig,"model_comparison")
    for study,name,xlabel in ((results["dataset_size_study"],"dataset_size","Training configurations"),
                              (results["ablation"],"ablation","Feature set")):
        keys=sorted(set(r["n_groups"] for r in study)) if name=="dataset_size" else list(dict.fromkeys(r["features"] for r in study))
        fig,ax=plt.subplots(figsize=(7.5,4.5),layout="constrained")
        positions=keys if name=="dataset_size" else np.arange(len(keys))
        for field,label in (("validation_rmse","Validation"),("test_rmse","Test (final evaluation)")):
            vals=[[r[field] for r in study if (r["n_groups"] if name=="dataset_size" else r["features"])==k and r["success"]] for k in keys]
            means=[np.mean(v) if v else np.nan for v in vals]; stds=[np.std(v,ddof=1) if len(v)>1 else 0 for v in vals]
            ax.errorbar(positions,means,yerr=stds,fmt="o-",capsize=4,label=label)
        if name=="ablation":
            ax.set_xticks(positions,["d","d + a","d + s","d + a + s"])
        ax.set(xlabel=xlabel,ylabel="RMSE (fraction)",title="Mean ± sample SD over converged fixed seeds")
        ax.legend(); finish(fig,name)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    measures=[timing["mc_api"],timing["mc_transport"],timing["models"]["mlp"]["single"]]
    values=[m["median_seconds"]*1e6 for m in measures]
    errors=np.array([[v-m["q25_seconds"]*1e6 for v,m in zip(values,measures)],
                     [m["q75_seconds"]*1e6-v for v,m in zip(values,measures)]])
    ax.bar(["MC API","MC transport","MLP single"],values,yerr=errors,capsize=4)
    ax.set(ylabel="Microseconds / configuration (log)",yscale="log",title="Matched configurations; median and IQR of 7 sweeps")
    finish(fig,"timing")
    fig,ax=plt.subplots(figsize=(7,4.5),layout="constrained")
    upper=timing["research_break_even"] or 10000
    queries=np.linspace(0,upper*1.3,200); t_pred=timing["models"]["mlp"]["single"]["median_seconds"]
    ax.plot(queries,queries*timing["mc_api"]["median_seconds"],label="Repeated warm MC")
    for key,label in (("deployment_setup_seconds","ML deployment setup"),("research_setup_seconds","ML research setup")):
        ax.plot(queries,timing[key]+queries*t_pred,label=label)
    ax.set(xlabel="Number of requests",ylabel="Accumulated seconds",title="Conditional break-even; unequal error structures")
    ax.legend(fontsize=9); finish(fig,"break_even")
    slice_rows=[r for r in diagnostics["rows"] if r["diagnostic"]=="slice_s1"]
    side=round(np.sqrt(len(slice_rows)))
    widths=sorted(set(r["thickness_cm"] for r in slice_rows)); absorptions=sorted(set(r["absorption_rate"] for r in slice_rows))
    grid=np.array([r["mlp_prediction"]-r["target"] for r in slice_rows]).reshape(side,side)
    bound=max(abs(grid.min()),abs(grid.max()))
    fig,ax=plt.subplots(figsize=(6.8,4.8),layout="constrained")
    mesh=ax.pcolormesh(widths,absorptions,grid.T,cmap="coolwarm",vmin=-bound,vmax=bound,shading="nearest")
    ax.scatter([r["thickness_cm"] for r in slice_rows],[r["absorption_rate"] for r in slice_rows],s=6,c="black")
    ax.set(xlabel="Thickness (cm)",ylabel="Absorption rate (1/cm)",title="Observed MC slice: scattering=1 /cm")
    # Cell shading must not imply an observed out-of-domain region.
    ax.set_xlim(FEATURE_SPEC_V2[0]["domain"])
    ax.set_ylim(FEATURE_SPEC_V2[1]["domain"])
    fig.colorbar(mesh,ax=ax,label="MLP − MC transmission"); finish(fig,"error_slice")
    fig,ax=plt.subplots(figsize=(6.8,4.2),layout="constrained")
    noise=diagnostics["noise_aggregate"]
    ax.loglog([r["histories"] for r in noise],[r["empirical_rms_sd"] for r in noise],"o-",label="Empirical RMS replicate SD")
    ax.loglog([r["histories"] for r in noise],[r["mean_plugin_se"] for r in noise],"s--",label="Mean binomial plug-in SE")
    ax.set(xlabel="Histories per run",ylabel="Uncertainty (fraction)",title="Independent repeated MC; fixed diagnostic configurations")
    ax.legend(); finish(fig,"noise")


def manifest_v2(directory, config):
    members={str(p.relative_to(directory)).replace("\\","/"):dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in directory.rglob("*") if p.is_file() and p.name!="manifest.json"}
    atomic_json(directory/"manifest.json",dict(version=VERSION_V2,source_hashes=config["source_hashes"],
        artifacts=members,authentication="integrity only; not author authentication"))


def report_tables_v2(directory):
    """Each report table is derived from saved artifacts, never typed metrics."""
    config=read_json(directory/"experiment_config.json"); result=read_json(directory/"metrics.json")
    timing=read_json(directory/"timing.json"); selection=read_json(directory/"selection.json")
    diagnostic=read_json(directory/"diagnostics.json"); model=load_model(directory/"mlp.json")
    environment=config["environment"]
    tables={}

    def add(number,title,source,rows):
        tables[str(number)]=dict(title=title,source=source,rows=rows)

    add(1,"Вычислительная система и программная среда","experiment_config.json: environment",
        [dict(parameter=k,value=v) for k,v in environment.items()])
    add(2,"Входные признаки и область применимости","experiment_config.json: feature_spec",
        [dict(position=s["position"],name=s["name"],unit=s["unit"],lower=s["domain"][0],upper=s["domain"][1],type=s["quantity_type"]) for s in feature_spec()])
    add(3,"Параметры основного Monte Carlo, все конфигурации","experiment_config.json; generation.json",
        [dict(parameter=k,value=config[k]) for k in ("histories_per_repeat","repeats","source_mu","source_x","max_collisions","backend")]
        +[dict(parameter="groups",value=len(config["groups"])),dict(parameter="total_histories",value=len(config["groups"])*config["repeats"]*config["histories_per_repeat"])])
    add(4,"Разбиение по независимым физическим группам","splits.json; experiment_config.json",
        [dict(part=part,groups=len(ids),runs=len(ids)*config["repeats"],histories=len(ids)*config["repeats"]*config["histories_per_repeat"]) for part,ids in config["partitions"].items()])
    rows=[]
    for candidate in config["candidates"]:
        matched=[r for r in selection["trials"] if r["widths"][1:-1]==candidate["hidden"] and r["alpha"]==candidate["alpha"]]
        rows.append(dict(hidden=str(candidate["hidden"]),alpha=candidate["alpha"],seeds=len(matched),
            converged=sum(r["success"] for r in matched),validation_rmse_mean=float(np.mean([r["validation"]["rmse"] for r in matched])),
            iterations_min=min(r["iterations"] for r in matched),iterations_max=max(r["iterations"] for r in matched)))
    add(5,"Архитектуры MLP; validation, три фиксированных seed","selection.json: trials; experiment_config.json: candidates",rows)
    add(6,"Сравнение моделей на неизменном test; ошибки в долях","metrics.json; comparison.csv",
        [dict(model=name,n_test=len(config["partitions"]["test"]),**result[name]["test"]) for name in ("dummy","ridge","polynomial_ridge","mlp")])
    add(7,"Устойчивость основной архитектуры по seed; test","metrics.json: seed_sensitivity",result["seed_sensitivity"])
    for number,key,source in ((8,"ablation","ablation.csv"),(9,"dataset_size_study","dataset_size_study.csv")):
        rows=[]
        study=result[key]; feature_keys=list(dict.fromkeys(r["features"] if key=="ablation" else r["n_groups"] for r in study))
        for value in feature_keys:
            matched=[r for r in study if (r["features"] if key=="ablation" else r["n_groups"])==value]
            # Failed fits remain visible; their finite errors are diagnostics,
            # not accepted successful replications.
            rows.append(dict(features=matched[0]["features"],n_groups=matched[0]["n_groups"],
                histories=matched[0]["training_histories"],converged=sum(r["success"] for r in matched),attempts=len(matched),
                validation_rmse_mean=float(np.mean([r["validation_rmse"] for r in matched])),
                test_rmse_mean=float(np.mean([r["test_rmse"] for r in matched])),
                test_rmse_sd=float(np.std([r["test_rmse"] for r in matched],ddof=1)),
                training_seconds_mean=float(np.mean([r["training_seconds"] for r in matched]))))
        add(number,"Вклад признаков, три seed" if number==8 else "Размер train при неизменных validation/test, три seed",source,rows)
    add(10,"Ошибка MLP по областям каждого признака; test","error_regions.csv",result["error_regions"])
    rows=[]
    for name,m in timing["models"].items():
        for operation,measurement in (("single_predict",m["single"]),("batch_predict",m["batch"]),("load",m["load"])):
            rows.append(dict(model=name,operation=operation,n_configurations=m["batch_size"] if operation=="batch_predict" else 1,
                median_seconds=measurement["median_seconds"],q25_seconds=measurement["q25_seconds"],q75_seconds=measurement["q75_seconds"]))
    for key in ("mc_api","mc_transport"):
        m=timing[key]; rows.append(dict(model="Monte Carlo",operation=key,n_configurations=len(config["partitions"]["test"]),
            median_seconds=m["median_seconds"],q25_seconds=m["q25_seconds"],q75_seconds=m["q75_seconds"]))
    add(11,"Время, секунды: совпадающие test-конфигурации, семь прогретых серий","timing.json",rows)
    add(12,"Условная окупаемость при усреднённой стоимости test-запроса","timing.json",
        [dict(accounting=label,setup_seconds=timing[setup],queries=timing[queries],conditional=True)
         for label,setup,queries in (("Deployment","deployment_setup_seconds","deployment_break_even"),
                                     ("Research","research_setup_seconds","research_break_even"))])
    add(13,"Шум Monte Carlo и независимые более точные метки","metrics.json: label_noise; diagnostics.json",
        [dict(quantity=k,value=v) for k,v in result["label_noise"].items() if k!="caveat"]
        +[dict(quantity="main_test_mlp_rmse",value=result["mlp"]["test"]["rmse"]),
          dict(quantity="independent_lhs_mlp_rmse",value=diagnostic["scores"]["independent_lhs"]["mlp"]["rmse"]),
          dict(quantity="independent_lhs_se_median",value=diagnostic["diagnostic_se_median"])]
        +[dict(quantity=f"replicate_sd_at_{r['histories']}_histories",value=r["empirical_rms_sd"]) for r in diagnostic["noise_aggregate"]])
    atomic_json(directory/"report_tables.json",tables)
    return tables


def markdown_table(table):
    rows=table["rows"]; keys=list(rows[0])
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return "да" if value else "нет"
        if isinstance(value,float):
            return f"{value:.8g}"
        return str(value).replace("|","/").replace("\n"," ")
    return "\n".join(["| "+" | ".join(keys)+" |","|"+"---|"*len(keys)]
        +["| "+" | ".join(cell(row.get(k)) for k in keys)+" |" for row in rows])


def verify_report_v2(directory, report_path):
    report=Path(report_path).read_text(encoding="utf-8")
    tables=read_json(directory/"report_tables.json")
    for number,table in tables.items():
        block="<!-- TABLE_"+number+"_BEGIN -->\n"+markdown_table(table)+"\n<!-- TABLE_"+number+"_END -->"
        if block not in report:
            raise ValueError("report table mismatch: "+number)
    result=read_json(directory/"metrics.json")
    comparison=list(csv.DictReader((directory/"comparison.csv").open(encoding="utf-8")))
    for row in comparison:
        for key in ("mae","rmse","r2","max_absolute_error","bias"):
            if float(row[key])!=result[row["model"]]["test"][key]:
                raise ValueError("comparison CSV mismatch")
    check=dict(status="passed",tables=len(tables),comparison_metrics_checked=20,
        report_sha256=hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
        sources={n:hashlib.sha256((directory/n).read_bytes()).hexdigest()
                 for n in ("metrics.json","comparison.csv","timing.json","selection.json","report_tables.json")},
        scope="exact generated table blocks and comparison.csv against metrics.json; narrative reviewed separately")
    atomic_json(directory/"report_consistency.json",check)
    return check


def research_report_v2(directory):
    """Russian research chapter; numerical tables are generated from artifacts."""
    import os
    tables=report_tables_v2(directory)
    config=read_json(directory/"experiment_config.json"); results=read_json(directory/"metrics.json")
    timing=read_json(directory/"timing.json"); model=load_model(directory/"mlp.json")
    diagnostics=read_json(directory/"diagnostics.json"); selection=read_json(directory/"selection.json")
    relative=os.path.relpath(directory,ROOT/"docs").replace("\\","/")
    blocks=[]

    def paragraph(value):
        blocks.append(value.strip()+"\n")

    def table(number):
        t=tables[str(number)]
        paragraph(f"**Таблица {number}. {t['title']}.** Источник: `{relative}/{t['source']}`.")
        paragraph(f"<!-- TABLE_{number}_BEGIN -->\n{markdown_table(t)}\n<!-- TABLE_{number}_END -->")

    def figure(name,caption):
        paragraph(f"![{caption}]({relative}/figures/{name}.png)\n\n{caption}")

    paragraph('''# Разработка открытого научного программного инструмента на Python для моделирования транспорта нейтронов методом Монте-Карло с использованием суррогатных моделей машинного обучения

## Многопараметрическая аппроксимация результатов Monte Carlo-моделирования методами машинного обучения

Исследовательский отчёт HeliosMC, версия 2. Дата выполнения: 19 ноября 2026 года.
Образовательный исследовательский проект РКНП.

Предыдущие одномерные модели и измерения сохранены в
`benchmarks/results/ml_baseline/full/`. Настоящий документ описывает
отдельный многопараметрический эксперимент `synthetic_slab_surrogate_v2`.
Одномерный отчёт доступен в истории Git: `d8c9d37:docs/ML_RESEARCH_REPORT.md`.
Значения v1 и v2 нельзя объединять: они относятся к разным постановкам и данным.

## Аннотация''')
    paragraph(f'''Исследуется возможность ускорить повторные обращения к образовательному
Monte Carlo-симулятору HeliosMC с помощью регрессионной модели, обученной на его
результатах. Задача определяется тремя входами: толщиной слоя и постоянными
коэффициентами поглощения и рассеяния синтетической среды. Использованы
{len(config['groups'])} независимых конфигураций Latin Hypercube Sampling, по три
MC-повтора каждой конфигурации. Сравниваются константный регрессор, линейная
Ridge-регрессия, полиномиальная Ridge второй степени и MLP на NumPy/SciPy.
Test RMSE выбранной MLP равна {results['mlp']['test']['rmse']:.8g}, полиномиальной
модели — {results['polynomial_ridge']['test']['rmse']:.8g} абсолютной доли.
При совпадающих тестовых входах отношение прогретого времени MC API к времени
одиночного MLP-predict составило {timing['speedup']:.5g}. Вычислительная
окупаемость оценивается условно: эквивалентная точность MC и MLP для каждого
запроса не установлена. Отдельно выполнены исследования состава признаков,
объёма train, MC-шума и независимого набора с повышенным числом историй.
Работа проверяет свойства эмулятора конкретного учебного алгоритма; инженерная
и экспериментальная валидация реального материала не проводилась.''')
    paragraph('''## 1. Введение

Метод Монте-Карло позволяет исследовать математическую модель через множество
случайных реализаций. В транспортной задаче каждая история представляет один
путь частицы со случайными расстояниями между взаимодействиями и случайными
исходами. Частота наблюдаемого исхода становится оценкой соответствующей
вероятности. Такой подход удобен, когда суммарное поведение складывается из
многих ветвящихся возможностей и прямое вычисление распределения затруднено.
Историческую постановку метода обсуждают Metropolis и Ulam [1].

Вычислительные затраты особенно заметны в серии запросов. Один расчёт может
занимать миллисекунды, однако при подборе учебных параметров, построении срезов
или повторном анализе приходится обращаться к модели многократно. Увеличение
числа историй уменьшает случайный разброс, но увеличивает стоимость каждого
обращения. Этот конфликт сохраняется даже после ускорения программного ядра.

Суррогатная модель строит приближение зависимости выхода симулятора от входов.
Сначала необходимо оплатить получение обучающих данных и настройку модели.
После этого новый ответ вычисляется прямым применением регрессора. Обзор Hong
и Zhang [2] описывает эту общую идею и различные семейства приближений. Само
применение MLP к результатам симуляции давно известно; оно не составляет
самостоятельного доказательства научной новизны настоящего проекта.

HeliosMC выбран как открытая образовательная среда с контролируемыми
допущениями. Программная модель достаточно компактна для проверки баланса
исходов, сохранения seed и исследования погрешностей. Учебная постановка
позволяет различать три вопроса: правильно ли работает программа, хорошо ли
регрессор воспроизводит её результаты и соответствует ли сама модель реальному
веществу. В данной работе проверяются первые два вопроса.

## 2. Постановка проблемы

Образовательная проблема состоит в том, что одиночный стохастический результат
легко принять за точный ответ. Небольшое расхождение двух расчётов может быть
обычным статистическим шумом. Аналогично, малую ошибку нейросети можно
переоценить, если в test попали повторные запуски уже известных конфигураций.
Проект должен делать происхождение данных, повторы и границы вывода видимыми.

Исследовательская проблема формулируется как компромисс между точностью,
обобщающей способностью и совокупной стоимостью вычислений. Нас интересует,
при каком составе признаков и числе обучающих конфигураций регрессор даёт
приемлемое приближение, а также сколько повторных запросов требуется для
амортизации подготовки. Быстрый ответ, не соответствующий нужной точности,
сам по себе не является вычислительным преимуществом для пользователя.

Объект исследования — существующая однородная slab-модель независимых историй
без размножения. Предмет — точность и стоимость её регрессионного приближения.
Работа не ставит целью создать инженерный аналог OpenMC, рассчитывать
критичность или переносить найденные коэффициенты на реальный материал.
Все изменяемые скорости взаимодействий заданы как синтетические учебные числа.

## 3. Цель, задачи и проверяемые гипотезы

Цель — построить воспроизводимый трёхпараметрический эксперимент, в котором
можно проверить условия полезности ML-суррогата. До основного обучения
зафиксированы области входов, метод выборки, критерий выбора модели, seed,
разбиение и исследования дополнительных факторов. Четвёртый признак source_mu
не включён: фиксированная постановка с тремя входами позволяет завершить
основное исследование без расширения его объёма.

H1: MLP точнее линейной модели на новых конфигурациях. Проверка — общий test
и парный бутстреп разности RMSE; дополнительно сравнивается полиномиальная
модель, способная описывать нелинейность и взаимодействия входов.

H2: увеличение числа train-групп в целом уменьшает ошибку до насыщения.
Проверка — вложенные подвыборки 20/40/60/80/100% при неизменном validation/test
и заранее выбранной архитектуре. Строгая монотонность каждой случайной кривой
не предполагается. Основной критерий сравнивает среднюю validation RMSE при
100% и 20%; по test оптимальный размер не выбирается.

H3: доступность разных входов неодинаково влияет на ошибку предиктора.
Проверка — одна толщина, толщина с поглощением, толщина с рассеянием, все три
признака. Это анализ информации, доступной конкретному регрессору, а не
оценка причинной значимости параметров реального вещества.

H4: при большом количестве запросов меньшая стоимость inference может
компенсировать подготовку модели. Проверяются знак разности времён и условная
точка окупаемости с двумя явно описанными способами учёта подготовки.

H5: шум MC-меток ограничивает интерпретацию ошибки ML. Проверка — независимые
повторы, несколько чисел историй и дополнительные конфигурации с более точными
MC-метками. Неопределённость самой физической модели этим не оценивается.

Основной учебный критерий до обучения: test RMSE ≤0,015 и максимальная ошибка
≤0,06 абсолютной доли. Эти пороги предназначены для данного исследования;
они не являются нормативами точности нейтронного транспорта.

## 4. Теоретическая основа

### 4.1. Независимые истории и статистическая оценка

Внутри слоя постоянны поглощение a и рассеяние s, оба в см⁻¹. Суммарная
скорость равна a+s. До следующего взаимодействия выбирается оптическое
расстояние τ=−ln(1−U), где U равномерно на [0,1). Геометрическое расстояние
в однородной среде равно τ/(a+s). Если раньше достигается граница, частица
выходит. При столкновении поглощение происходит с вероятностью a/(a+s),
иначе новое направление выбирается по существующему изотропному закону.
Энергия не изменяется; коэффициенты после рассеяния остаются прежними.

Пусть K из N завершённых историй вышли справа. Тогда p̂=K/N — оценка
вероятности этого исхода. При независимых историях K имеет биномиальную
модель, а plug-in стандартная ошибка равна sqrt(p̂(1−p̂)/N). Она характеризует
разброс оценок при повторении вычисления. Это не гарантированная граница
ошибки отдельного запуска и не неопределённость исходных коэффициентов.
Используемые интервалы Clopper–Pearson вычисляет существующая функция
summarize_binomial; описание биномиальных методов приведено в SciPy [9].

Полный выход справа включает рассеянные истории. При s>0 выражение
exp(−(a+s)d) соответствует только выходу без столкновений и не используется
как эталон полного прохождения. В отдельном предельном случае s=0,
нормальном входе и x=0 вероятность полного прохождения равна exp(−ad).
Этот случай позволяет проверять математическую реализацию независимо от ML.

Есть дополнительное свойство самой постановки: при фиксированном нормальном
входе её безразмерное поведение определяется оптическими произведениями ad
и sd. Одновременное изменение d, a и s, сохраняющее эти произведения,
даёт математически эквивалентную транспортную задачу. Следовательно, три
независимо выбираемых размерных входа не означают три независимых
безразмерных механизма. В основной модели сохранены именно запрошенные
исходные признаки; сравнение с регрессией по оптическим параметрам остаётся
отдельной возможностью дальнейшего исследования.

### 4.2. Регрессионные модели

Dummy предсказывает среднее y по train для любого входа. Если более сложная
модель не улучшает его результат, полезная зависимость не выявлена.
Линейная модель использует стандартизированные входы z и предсказывает
b+Σβⱼzⱼ. Ridge добавляет к сумме квадратов ошибок штраф αΣβⱼ²; свободный член
не штрафуется. В программе задача решается расширенным методом наименьших
квадратов, что допускает зависимые и постоянные столбцы без обращения
плохо обусловленной матрицы нормальных уравнений.

Полиномиальная модель второй степени использует z₁,z₂,z₃, их квадраты и
попарные произведения. Вместе со свободным членом получается десять
коэффициентов. Такая модель может описывать кривизну и взаимодействия
параметров при небольшой сложности. Сравнение с ней помогает избежать
преувеличения преимуществ MLP над слишком слабым линейным ориентиром.

MLP последовательно вычисляет hₗ=tanh(hₗ₋₁Wₗ+bₗ) в скрытых слоях и линейный
выход. Число входов берётся из схемы модели. Обучаемая цель:
L=mean((ŷ−y)²)/2+αΣ||Wₗ||²/2. Смещения не регуляризуются. Target остаётся
в исходных единицах доли. Ограничение ŷ∈[0,1] не навязывается, поэтому
выходы за границы измеряются явно и не скрываются clipping.

Аналитический backpropagation применяет цепное правило к этой функции.
Для tanh производная равна 1−h². Численный тест сравнивает весь вектор
градиента с конечными разностями в сетях с тремя входами, одним и двумя
скрытыми слоями. Успех такой проверки подтверждает согласованность кода
дифференцирования; он не устанавливает качество данных или полноту модели.

L-BFGS-B аппроксимирует информацию о кривизне, сохраняя ограниченную историю
изменений параметров. Оптимизатор получает значение цели и аналитический
градиент, выполняет поиск шага и проверяет условия остановки [4]. Его
success означает выполнение численного критерия, а не доказательство
глобального минимума. Достижение maxiter сохраняется как неуспешный исход.
Встроенной случайной validation-выборки у нашего fit нет: все группы
validation заданы вне оптимизатора.

### 4.3. Обобщение, метрики и интервалы

Стандартизация вычисляет среднее и стандартное отклонение каждого столбца
только по train. Постоянство определяется совпадением минимума и максимума;
его масштаб заменяется единицей, чтобы исключить деление на округлённый ноль.
Сохранённая модель содержит scaler и точный порядок признаков. Для числовой
матрицы порядок столбцов является контрактом вызывающего кода; по самим
числам распознать произвольную перестановку невозможно. Для именованного
ввода порядок проверяется явно.

На отложенных группах вычисляются MAE=mean(|ŷ−y|),
RMSE=sqrt(mean((ŷ−y)²)), максимальная абсолютная ошибка и Bias=mean(ŷ−y).
R²=1−Σ(ŷ−y)²/Σ(y−ȳ)² может быть отрицательным, если предиктор хуже
выборочного среднего. Доля ответов вне [0,1] хранится отдельно от их числа.
При постоянном y R² не определён и сохраняется как null. Формулы согласуются
с официальным описанием метрик [6]; библиотека scikit-learn не используется
в вычислительном pipeline и не устанавливалась.

Бутстреп повторно выбирает тестовые конфигурации с возвращением и оценивает
распределение метрик [3]. Все повторы одной конфигурации сначала объединены;
они не считаются независимыми наблюдениями для бутстрепа. Для разности RMSE
двух моделей применяются одни и те же bootstrap-индексы. Полученные интервалы
условны на обученных моделях, единственном разбиении и шумных MC-метках.
Они не включают всю изменчивость переобучения, проектирования выборки или
ошибку исходных математических допущений.

## 5. Архитектура HeliosMC

`material.py` задаёт MaterialDataset, единицы и проверки коэффициентов.
`physics.py` содержит simulate_slab, агрегированные статистики и родной
экспорт эксперимента. `numba_compat.py` предоставляет Numba/fallback.
`benchmarks/ml_baseline.py` вызывает эти API, строит датасет, обучает модели,
оценивает их и создаёт результаты. `tests/test_ml_baseline.py` проверяет
численную и программную корректность исследования. Существующие `gui.py`,
`simulation_worker.py` и `main.py` отвечают за интерактивное приложение;
обучение не выполняется в GUI или при импорте исследовательского модуля.

```mermaid
flowchart LR
    A[Именованные входные параметры] --> B[Синтетический MaterialDataset]
    B --> C[simulate_slab / Monte Carlo]
    C --> D[Проверенный экспорт и dataset]
    D --> E[Разбиение по группам]
    E --> F[Scaler по train]
    F --> G[Dummy / Ridge / Polynomial / MLP]
    G --> H[Validation: выбор]
    H --> I[Test, диагностика, время и отчёт]
```

Модернизация повторно использует forward, objective и существующие API
экспорта. Размер входного слоя теперь задаётся схемой. Формат модели v2
отличается от v1; сохранённые водные модели не переименовываются в
трёхпараметрические. Загрузка проверяет единицы, порядок, области,
размеры scaler, веса, смещения и конечность чисел. JSON содержит данные;
pickle не применяется. Старые модели v1 по-прежнему читаются по их
собственной схеме и принимают только толщину.

Исходные bundles содержат experiment.json, results.csv и arrays.npz.
Они упакованы в ZIP без изменения внутренней схемы. При чтении извлекаются
только фиксированные допустимые члены во временный каталог и вызывается
родной валидатор. SHA256 защищает целостность, но не аутентифицирует автора.
Исходные MC-записи и агрегированные группы сохранены отдельно, чтобы
не потерять seed, счётчики, время и оценку неопределённости.

## 6. Методология

### 6.1. Окружение, признаки и фиксированные условия''')
    table(1)
    paragraph('''Числа времени ниже относятся к этому окружению и его текущей нагрузке.
Потоки BLAS ограничены переменными процесса; GPU не требуется. Точная модель
ноутбука и его номинальный объём RAM не определялись, поэтому они не указаны.
Массивы обучения содержат сотни или тысячи строк, что ограничивает память;
общий RSS процесса в этой работе отдельно не измерялся.''')
    table(2)
    paragraph('''Диапазоны зафиксированы до пилота и test. Положительное поглощение
снижает риск очень долгих почти консервативных историй. Максимальная
суммарная оптическая толщина равна 9,6. Нулевое рассеяние допускается как
проверяемая граница, а положительные значения создают разнообразные пути
выхода. Эти области выбраны для численной управляемости и разнообразия
учебного отклика; они не являются диапазонами свойств какого-либо вещества.
Пилот проверяет пригодность фиксированной области, но не корректирует её
по неудобным результатам независимого тестирования.''')
    table(3)
    paragraph('''Источник фиксирован на левой границе и направлен нормально внутрь
слоя. Ядерные сечения, плотность реального вещества, температура и энергия
не выступают входами этой синтетической задачи. Лимит столкновений является
вычислительной защитой; исчерпанные или численно незавершённые истории
не превращаются в допустимые метки. Каждое наблюдение проверяется на
баланс счётчиков, положительное число историй, конечность статистик,
совместимость материала и соответствие доли сохранённым числителю и знаменателю.

### 6.2. Проектирование данных и статистические повторы

Рассмотрены независимая равномерная выборка, структурированная сетка и LHS.
Независимые случайные точки просты, но могут оставлять незаполненные интервалы
каждого входа. Полная сетка удобна для двумерного среза, однако число её точек
растёт как произведение числа уровней по координатам. LHS стратифицирует
каждое одномерное распределение, не требуя полного декартова произведения [5].
Выбран scrambled LHS без оптимизации после выборки; его seed сохранён.
Сравнение вариантов проектирования здесь методологическое: их влияние на
ошибку не измерялось в отдельном конкурентном исследовании.

Пилот включает 15 конфигураций и три повтора по 2000 историй. Есть центральная
точка, изменения каждого входа к двум границам и дополнительные LHS-точки.
Различие крайних откликов проверяется относительно объединённой MC SE.
Это подтверждает различимость эффектов при выбранном числе историй,
не устанавливая причинной важности признаков в обученной сети.

Объём основного набора выбирается только по стоимости пилота и заданному
бюджету: quick — 60 групп по три запуска с 1000 историями, full — от 300
до 900 групп по три запуска с 8000 историями. При недостаточном бюджете
минимальный дизайн не запускается. Временной предел относится к отдельному
вызову основной генерации; конечные подборы моделей и диагностические
вычисления имеют отдельный фиксированный объём. Уже записанные полные
bundles могут использоваться при возобновлении с теми же хешами источников.

Группа задаётся существующим group_id, включающим все три параметра
математической конфигурации и источник, но исключающим seed, число историй
и вычислительный backend. Полный configuration_id каждого отдельного
запуска дополнительно сохранён в исходном bundle. Повторы используют
различные seed. Target пула равен сумме успешных исходов, делённой на
сумму историй. Вместе с ним хранятся Clopper–Pearson CI, plug-in SE и
эмпирическая SEM независимых повторов. Записи не очищаются подстановкой
удобных чисел; ошибка генерации записывается и останавливает стадию.''')
    table(4)
    paragraph('''Разбиение выполняется по уникальным группам до обучения и не
использует target. Нормализация каждого fit обучается только на его train,
включая отдельные fit на уменьшенных наборах. Validation и test остаются
одними и теми же для сравнительных исследований. Test передаётся только
в этап окончательной оценки: функция подбора обучается даже если вместо
test-данных ей передан недоступный объект, что проверяет unit-тест.

### 6.3. Выбор моделей и дополнительные исследования

Ridge и polynomial Ridge сравниваются по α=0, 0,01 и 1 на validation.
Четыре MLP-настройки проверяются на фиксированных seed 17, 29 и 43.
Основной кандидат должен сойтись на всех трёх seed; среди допустимых
выбирается минимальное среднее validation RMSE. Основная сохранённая
модель всегда имеет seed 17, что предотвращает выбор самого удачного seed.
Сходимость других попыток, число итераций и причина остановки сохраняются.

После выбора архитектуры выполняются вложенные размеры train и ablation.
Архитектура, штраф и алгоритм остаются общими; число входных нейронов
соответствует подмножеству признаков. Для полной выборки и всех входов
повторно используются уже обученные модели: идентичный fit не считается
новым независимым экспериментом. В неполных ablation-моделях неоднозначность
ответа возникает от скрытых входов, даже если оптимизация сошлась идеально.

Диагностический набор с другим seed LHS не пересекается с основными
конфигурациями и содержит повышенное число историй. Отдельная наблюдаемая
сетка фиксирует s=1 см⁻¹ и меняет d,a. На её рисунке показаны реальные
MC-метки, а не вымышленные данные плотной тепловой карты. В отдельном
noise-study несколько фиксированных конфигураций рассчитываются при
трёх числах историй, по восемь повторов. Эти данные не используются для
настройки или повторного выбора модели.

### 6.4. Измерение времени

Монотонный perf_counter измеряет API Monte Carlo, отдельный transport timer,
одиночный predict, пакет, загрузку JSON и подготовку материалов/признаков.
Для основного сравнения используются те же test-входы обеих моделей.
Сначала прогревается ядро; затем выполняются семь одинаково устроенных
серий. Внутри каждой MC-серии время усредняется по test-конфигурациям,
а медиана и IQR вычисляются между сериями. Это не распределение стоимости
одного произвольного входа; неоднородность входов сохранена отдельно.

В single-predict включены проверка схемы и области, нормализация, подготовка
численных весов из сохранённого объекта и прямой проход. При batch сообщается
время всего пакета и его размер. Загрузка файла измеряется отдельно.
Материалы для MC заранее созданы; экспорт, GUI и рисунки в чистые timing-серии
не входят. Первый wrapper-вызов пилота в свежем процессе содержит JIT и
инициализацию, поэтому приводится отдельно от прогретого сравнения.

## 7. Результаты

### 7.1. Подбор и проверка обучения''')
    table(5)
    paragraph(f'''Выбранная архитектура: {model['widths']}, alpha={model['alpha']};
основной seed {model['seed']}. Оптимизатор выполнил {model['iterations']}
итераций, причина остановки: `{model['message']}`. Максимальный компонент
градиента {model['gradient_max']:.8g}. Численный статус не заменяет test-оценку.
Все попытки, включая несошедшиеся, доступны в selection.json и сохранённом
checkpoint обучения. При расчёте таблицы 5 средние ошибки показаны для всех
попыток; статус допуска к выбору читается по столбцу converged.

RMSE основной модели: train {results['mlp']['train']['rmse']:.8g}, validation
{results['mlp']['validation']['rmse']:.8g}, test {results['mlp']['test']['rmse']:.8g}.
Соотношение этих величин позволяет оценить переобучение, но один случайный
split не характеризует его полную изменчивость. Повторный fit при одинаковых
данных и seed дал максимальную разность validation-предсказаний
{selection['reproducibility_max_difference']:.8g}. Совпадение проверено в данном
окружении; межплатформенная побитовая воспроизводимость не утверждается.''')
    figure("learning_curve","Рисунок 1. История RMSE на train и validation; test не отслеживается. Регуляризованная loss отдельно сохранена в learning_curves.csv.")
    paragraph("### 7.2. Независимое сравнение моделей")
    table(6)
    paragraph('''Таблица сопоставляет модели на одном и том же наборе физических
конфигураций. Полиномиальная строка важна: она показывает, достаточно ли
второй степени для наблюдаемой зависимости, прежде чем объяснять результат
особым преимуществом нейросети. Отрицательный R² или выход за область
вероятностей сохраняется как результат. Для принятия учебного критерия
используются одновременно RMSE и максимальная ошибка, а не только R².''')
    interval=results["bootstrap"]["intervals"]
    paragraph(f'''Для MLP bootstrap-интервал RMSE: {interval['mlp']['rmse']};
MAE: {interval['mlp']['mae']}; Bias: {interval['mlp']['bias']}.
Парный 95%-интервал разности RMSE MLP−Ridge:
{interval['mlp_minus_ridge']}; MLP−Polynomial Ridge:
{interval['mlp_minus_polynomial_ridge']}. Отрицательная разность означает
меньшую ошибку MLP. Интервалы рассчитаны на 2000 повторных выборках
конфигураций с seed 4815. Это условное статистическое сравнение обученных
моделей, не доказательство общего превосходства алгоритма на любых данных.
Результат проверки заранее заданного учебного критерия:
**{'выполнен на основном test' if results['accuracy_pass'] else 'не выполнен на основном test'}**.''')
    figure("predicted_actual","Рисунок 2. MLP и pooled MC на новых test-конфигурациях. Прямая соответствует идеальному совпадению.")
    figure("model_comparison","Рисунок 3. RMSE четырёх моделей с 95%-интервалами бутстрепа по конфигурациям.")
    table(7)
    paragraph('''В таблице 7 все seed были заданы до обучения. Основной ответ не
усредняется по трём сетям, и наиболее удачный seed не выбирается по test.
Наблюдаемый разброс характеризует лишь три локальные оптимизации одной
архитектуры; он не охватывает другие разбиения или способы генерации данных.''')
    paragraph("### 7.3. Вклад признаков и число обучающих конфигураций")
    table(8)
    paragraph('''Потеря входа заставляет сеть приближать усреднённый ответ при
неизвестных значениях исключённых параметров. Поэтому рост ошибки не следует
трактовать как экспериментальное доказательство причинного влияния фактора
в реальном материале. При сравнении строк нужно также смотреть на converged:
ошибки несошедшихся fit остаются диагностическими. Все средние таблиц 8–9
включают выполненные попытки, а столбец converged показывает их численный
статус; на графиках для валидных моделей используются только сошедшиеся fit.''')
    figure("ablation","Рисунок 4. Ablation: d — толщина, a — поглощение, s — рассеяние. Среднее ± SD по сошедшимся seed; независимый test оценивается после обучения всех вариантов.")
    table(9)
    paragraph('''Подвыборки вложены друг в друга по одной фиксированной перестановке
train-групп. Каждая использует собственный scaler. Изменение test-ошибки
показано после фиксации протокола и не использовано для выбора размера.
Немонoтонность допустима из-за ограниченности данных, шума и локальной
оптимизации. По одному небольшому улучшению нельзя надёжно определить
универсальный оптимальный объём; оно требует повторных проектов выборки.''')
    figure("dataset_size","Рисунок 5. RMSE от числа train-конфигураций; validation и test фиксированы. Полосы — SD трёх сошедшихся seed, а не доверительный интервал по всем источникам неопределённости.")
    paragraph("### 7.4. Распределение ошибок в пространстве входов")
    table(10)
    paragraph('''Каждый вход разделён на три заранее одинаковых по ширине интервала;
остальные координаты в этих строках не фиксируются. Число test-точек в
интервале указано в n. Сравнение локальных RMSE описывает наблюдаемый набор,
но не даёт гарантии на каждую точку внутри ячейки. Отдельный двумерный срез
ниже действительно фиксирует рассеяние и использует дополнительные MC-расчёты.''')
    figure("residual_distribution","Рисунок 6. Распределение test-остатков MLP−MC; clipping не применяется.")
    for i,spec in enumerate(feature_spec(),7):
        figure("residual_"+spec["name"],f"Рисунок {i}. Остаток по {spec['label'].lower()}, {spec['unit']}. Полосы ±1,96 MC SE относятся только к шуму метки.")
    figure("error_slice","Рисунок 10. Ошибка на отдельном вычисленном MC-срезе при s=1 см⁻¹. Чёрные точки — реально рассчитанные конфигурации; между ними нет дополнительных MC-наблюдений.")
    paragraph("### 7.5. Скорость и условная окупаемость")
    table(11)
    paragraph(f'''Одиночный прогретый MLP-predict имеет медиану
{timing['models']['mlp']['single']['median_seconds']:.8g} с; MC API на тех же
test-входах — {timing['mc_api']['median_seconds']:.8g} с. Отношение составляет
{timing['speedup']:.6g}; отношение transport timer к полному predictor —
{timing['transport_speedup']:.6g}. Это ускорение одного обращения к уже
загруженной модели. Первый wrapper-вызов пилота занял {timing['cold_call']:.8g} с
с возможной JIT-компиляцией и инициализацией. Он не участвует в этом отношении.
Генерация основных данных с экспортом заняла {timing['generation']['wall_seconds']:.8g} с,
последующее проверочное чтение/агрегация — {timing['generation']['aggregation_seconds']:.8g} с.
Сумма MC API-времён — {timing['generation']['simulation_seconds']:.8g} с.
Поиск гиперпараметров — {timing['search_training_seconds']:.8g} с,
дополнительные fit исследований — {timing['study_training_seconds']:.8g} с.
Различие этих затрат показывает, почему скорость прямого прохода недостаточна
для оценки стоимости всей работы.''')
    figure("timing","Рисунок 11. Прогретые измерения на совпадающих test-конфигурациях; медиана и IQR семи серий, логарифмическая шкала времени.")
    paragraph('''Пусть S — стоимость подготовки, t_MC — среднее время запроса к MC,
а t_ML — время запроса к загруженному predictor. Тогда накопленные затраты
равны T_MC(N)=N·t_MC и T_ML(N)=S+N·t_ML. При t_MC>t_ML их пересечение:
N*=S/(t_MC−t_ML). Если разность неположительна, конечной окупаемости по
этому способу расчёта нет; такое поле сохраняется как null.

Deployment включает сумму MC API-времён основного датасета, нормализацию,
один выбранный fit и одну загрузку модели. Это нижняя вычислительная оценка:
она не включает все операции хранения и контроля. Для консервативности
оплачиваются метки всех трёх частей датасета, а не только train.
Research включает wall time пилота и генерации с экспортом, проверочное
чтение, все поисковые fit, повторный fit воспроизводимости, дополнительные
ablation/size-fit и отдельные диагностические вычисления. Время самого
timing-бенчмарка, графиков, отчёта, тестов и работы человека не включено.
Следовательно, research-оценка расширенная, но не полная стоимость проекта.''')
    table(12)
    paragraph('''В обоих вариантах предполагаются повторные запросы из похожего
распределения и многократное использование модели в памяти. Численная
приемлемость среднего test RMSE не доказывает эквивалентность ошибок MC и MLP
для каждого запроса. Поэтому таблица 12 описывает условную амортизацию времени,
а не установленную равноточную замену транспортного расчёта. Полная серия
request может иметь другие толщины и число столкновений, что изменит порог.''')
    figure("break_even","Рисунок 12. Условные накопленные затраты; кривые построены по измеренным временам, эквивалентная точность не установлена.")
    paragraph("### 7.6. MC-шум и независимая диагностика")
    table(13)
    paragraph(f'''Дополнительный независимый LHS содержит {config['studies']['diagnostic_lhs']}
конфигураций с {config['studies']['diagnostic_histories']} историями в каждой.
Его MLP RMSE равна {diagnostics['scores']['independent_lhs']['mlp']['rmse']:.8g},
медианная SE меток — {diagnostics['diagnostic_se_median']:.8g}. Этот набор
помогает оценить ошибку относительно более точных, но всё ещё случайных меток.
Он генерируется после замораживания моделей; его результаты не возвращаются
в обучение. Различие с обычным test одновременно зависит от меньшего шума и
других конфигураций, поэтому не может быть целиком приписано изменению N.

Noise-study сохраняет все восемь seed на каждом числе историй, а также
эмпирический SD и среднюю plug-in SE. Для независимых биномиальных историй
ожидается масштаб N^−1/2, но восемь повторов дают заметно случайную оценку
SD. Более точная MC-метка не устраняет смещение аппроксимации нейросети.
Отдельные проверки s=0 сравниваются с exp(−ad) через уже существующий
предварительно заданный exact-binomial протокол. Их численные результаты
и статусы сохранены в diagnostics.json, а не заменены заявлениями о
физической валидации.''')
    figure("noise","Рисунок 13. Изменение эмпирического разброса повторных MC-оценок и биномиальной SE с числом историй; набор конфигураций фиксирован.")
    paragraph("## 8. Обсуждение и оценка гипотез")
    h1=interval["mlp_minus_ridge"][1]<0
    poly=interval["mlp_minus_polynomial_ridge"][1]<0
    size=tables["9"]["rows"]; small=min(size,key=lambda r:r["n_groups"]); large=max(size,key=lambda r:r["n_groups"])
    h2=large["validation_rmse_mean"]<small["validation_rmse_mean"]
    ab=tables["8"]["rows"]; full=ab[-1]["validation_rmse_mean"]
    h3=all(r["validation_rmse_mean"]>full for r in ab[:-1])
    noise_ratio=results["mlp"]["test"]["rmse"]/results["label_noise"]["pooled_se_rms"]
    diagnostic_ratio=diagnostics["scores"]["independent_lhs"]["mlp"]["rmse"]/diagnostics["diagnostic_se_median"]
    noise_first,noise_last=diagnostics["noise_aggregate"][0],diagnostics["noise_aggregate"][-1]
    noise_reduction=noise_first["empirical_rms_sd"]/noise_last["empirical_rms_sd"]
    expected_reduction=np.sqrt(noise_last["histories"]/noise_first["histories"])
    paragraph(f'''H1: парное сравнение с линейной моделью поддерживает гипотезу:
**{'да' if h1 else 'нет'}**. Аналогичное сравнение с полиномиальной моделью поддерживает
преимущество MLP: **{'да' if poly else 'нет'}**. Эти решения следуют из верхней границы
95%-интервала разности RMSE. Существенно, что полином учитывает взаимодействия;
его результат делает сравнение содержательнее, чем исходный одномерный
эксперимент только с линейной зависимостью.

H2: среднее validation RMSE при полном train меньше, чем при 20%: **{'да' if h2 else 'нет'}**.
Значения {large['validation_rmse_mean']:.8g} и {small['validation_rmse_mean']:.8g}
получены на одном фиксированном validation. Если промежуточные ступени не
монотонны, это не скрывается. Одной последовательности вложенных подвыборок
недостаточно для строгого вывода о точке насыщения или оптимальном объёме.

H3: все три заранее определённых сокращения признаков дают большую среднюю
validation RMSE, чем полный вход: **{'да' if h3 else 'нет'}**. Степень ухудшения различается
между строками таблицы 8. Важность зависит от выбранного распределения входов,
архитектуры и доступности остальных параметров. В однородной математической
модели также есть безразмерная избыточность: ad и sd могут быть более
экономным представлением, что пока не исследовано в конкурентном fit.

H4: разность прогретых времён положительна:
**{'да' if timing['deployment_break_even'] is not None else 'нет'}**. Условные пороги приведены
в таблице 12. Это подтверждает возможность амортизировать измеренные
вычислительные затраты при заданных предпосылках, но не устанавливает
практическую экономию при общем требовании к точности.

H5: отношение test RMSE к RMS биномиальных SE равно {noise_ratio:.6g};
на независимом более точном LHS отношение RMSE к медианной SE равно
{diagnostic_ratio:.6g}. Эмпирический RMS разброса при увеличении числа
историй от {noise_first['histories']} до {noise_last['histories']} уменьшился
в {noise_reduction:.6g} раза; ожидаемое соотношение по закону N^−1/2 равно
{expected_reduction:.6g}. Эти числа подтверждают наличие статистического
ограничения меток, но не позволяют приписать ему весь остаток MLP.
Чем сильнее RMSE превышает SE, тем заметнее дополнительная ошибка
аппроксимации/обобщения. Сопоставимость величин затрудняет её отделение от
MC-шума. Универсальный вывод о шуме как единственном или основном
ограничении не принимается; сравнение двух наборов также зависит от входов.

Нелинейный характер отклика объясняется тем, что d,a,s совместно определяют
не только прямое ослабление, но и длины случайных путей и вероятности
повторного изменения направления. Полином второй степени ограничивает
форму зависимости, а MLP допускает более гибкую композицию. Эта гибкость
увеличивает стоимость обучения и риск локальных минимумов. Простая модель
может быть предпочтительна, если её ошибка уже удовлетворяет задаче,
поскольку время fit, размер артефакта и интерпретируемость тоже имеют значение.

Скорость оправдывает применение только после определения приемлемой ошибки.
Для иллюстративного исследования параметров может быть достаточно среднего
учебного допуска; для решения с жёсткой гарантией по каждому входу этого
недостаточно. У MLP здесь нет проверенных предсказательных интервалов.
Расхождение на новых комбинациях, особенно у краёв области, должно оцениваться
независимыми MC-расчётами; обычный inference отказывает за пределами схемы.''')
    paragraph(f'''### 8.1. Наблюдаемые нарушения и предел учебного критерия

На основном test {results['mlp']['test']['outside_fraction']} из
{len(config['partitions']['test'])} ответов MLP оказались вне [0,1]; доля
таких ответов равна {results['mlp']['test']['out_of_range_fraction']:.8g}.
Выполнение порогов RMSE и Max Error на этом test не означает выполнения
всех требований к вероятностному предиктору. Линейный выход допускает
некорректные вероятности; clipping сознательно не применялся при оценке.

На отдельном более точном LHS максимальная ошибка достигла
{diagnostics['scores']['independent_lhs']['mlp']['max_absolute_error']:.8g};
на наблюдаемом срезе s=1 см⁻¹ RMSE равна
{diagnostics['scores']['slice_s1']['mlp']['rmse']:.8g}, максимальная ошибка —
{diagnostics['scores']['slice_s1']['mlp']['max_absolute_error']:.8g}.
Эти независимые результаты нужно сопоставлять с исходными порогами 0,015
и 0,06. Поэтому принятие основного test не даёт гарантии приемлемой ошибки
по всей заявленной области и не подтверждает равноточную замену MC.

В этой постановке отношение RMSE к MC SE и наблюдаемые локальные отклонения
показывают, что одного увеличения числа историй недостаточно для устранения
ошибки суррогата. Причины необходимо искать также в аппроксимации и обучении.
Скорость inference не устраняет эти нарушения; применение модели оправдано
лишь в задаче с явно допустимыми ошибками и большим числом повторных запросов.
Источник этих чисел — metrics.json и diagnostics.json; дополнительной
настройки модели после их просмотра не выполнялось.''')
    paragraph('''## 9. Ограничения исследования

Синтетические коэффициенты не являются измеренными сечениями реального
вещества. Модель не переносит энергию, не учитывает связанные состояния,
спектральные изменения и микроскопическую структуру материала. Проверка
предельного случая подтверждает математическую согласованность программы,
но не заменяет сравнение с экспериментом или независимо валидированным кодом.

Исследовано одно распределение LHS, одно основное разбиение и три seed
обучения. Полученные доверительные интервалы не отражают вариацию между
проектами выборки. Уменьшенные train-наборы вложены и статистически зависимы;
ошибки на последовательных ступенях нельзя интерпретировать как независимые
эксперименты. Диагностическая сетка двумерна и фиксирует только один уровень s.

Биномиальные SE и CI охватывают случайный разброс независимых историй.
Источник коэффициентов, ошибка математического приближения, оптимизация
и численная арифметика имеют иной смысл. Три MC-повтора основного набора
дают ограниченную эмпирическую оценку SEM. Бутстреп выполняется по группам,
но использует уже шумные target и не создаёт точного ground truth.

Сравнение алгоритмов ограничено четырьмя семействами и небольшим числом
гиперпараметров. Интерполяция, регрессия по оптическим параметрам, Gaussian
process и другие методы не тестировались. Их возможное преимущество или
проигрыш нельзя утверждать по настоящим результатам. Наличие трёх входов
само по себе не делает MLP необходимой.

Времена зависят от CPU, фоновой нагрузки, версий библиотек и размера пакета.
Ограничение потоков не гарантирует неизменной частоты процессора.
Расчёт break-even предполагает известное распределение будущих запросов и
не включает время человека. Программа не выдаёт инженерной гарантии
качества вне обученного домена, а сведения о производительности не являются
экономическим обоснованием применения в реальных ядерных установках.

## 10. Заключение''')
    paragraph(f'''Выполнено воспроизводимое исследование трёх независимо задаваемых
учебных входов. На {len(config['partitions']['test'])} отложенных конфигурациях
выбранная MLP получила MAE {results['mlp']['test']['mae']:.8g}, RMSE
{results['mlp']['test']['rmse']:.8g}, R² {results['mlp']['test']['r2']:.8g} и
максимальную ошибку {results['mlp']['test']['max_absolute_error']:.8g}.
Численные сравнения, ablation, размеры train и MC-повторы сохранены
раздельно и доступны для независимой проверки. Ускорение прогретого
одиночного обращения составило {timing['speedup']:.6g} при обозначенной
структуре ошибки и фиксированном распределении входов.

Практический результат состоит в проверяемой методике: пригодность
суррогата определяется требуемой точностью, полнотой входов, доступным
числом обучающих групп и повторным использованием модели. Полученные
пороги времени условны; физическая валидность и равноточная инженерная
замена MC не установлены. Следующее исследование целесообразно посвятить
сравнению с оптическими признаками и интерполяцией, повторным разбиениям,
более точным меткам и контролю ошибки у границ.

Использование NumPy, SciPy или MLP не заявляется как новизна. Самостоятельный
вклад проекта — реализация воспроизводимого образовательного инструмента
и конкретный экспериментальный анализ компромиссов на его математической
модели. Утверждение о приоритете относительно литературы здесь не делается.''')
    paragraph('''## 11. Литература

Источники проверены 19 ноября 2026 года. Для официальной документации указан
URL, а не вымышленный автор или год публикации.

1. Metropolis N., Ulam S. (1949). *The Monte Carlo Method*. Journal of the
American Statistical Association, 44(247), 335–341.
[DOI: 10.1080/01621459.1949.10483310](https://doi.org/10.1080/01621459.1949.10483310).
2. Hong L. Jeff, Zhang Xiaowei (2021). *Surrogate-Based Simulation Optimization*.
arXiv:2105.03893, версия 2. [Текст и библиография](https://arxiv.org/abs/2105.03893).
[DOI: 10.48550/arXiv.2105.03893](https://doi.org/10.48550/arXiv.2105.03893).
3. Efron B. (1979). *Bootstrap Methods: Another Look at the Jackknife*.
The Annals of Statistics, 7(1), 1–26.
[DOI: 10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552).
4. SciPy. *minimize(method='L-BFGS-B')*: условия остановки и статус оптимизации.
[Официальная документация](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
5. SciPy. *scipy.stats.qmc.LatinHypercube*: описание LHS и ссылки на исходные
публикации по планированию эксперимента.
[Официальная документация](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.qmc.LatinHypercube.html).
6. scikit-learn. *Metrics and scoring: regression metrics*.
[Официальная документация определений](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics).
Библиотека используется как справочный источник; вычисления реализованы на NumPy.
7. NumPy. *numpy.linalg.lstsq*: решение задачи наименьших квадратов.
[Официальная документация](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
8. Matplotlib. *Figure.savefig*: сохранение научных рисунков и разрешение.
[Официальная документация](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html).
9. SciPy. *BinomTestResult.proportion_ci*: биномиальные интервалы.
[Официальная документация](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

## 12. Приложения

### 12.1. Настройки, seed и схемы

Полный frozen-план находится в experiment_config.json. Планирование и split:
seed 20261119; основной MC: 800000+3·i+r; subset train: 1942;
bootstrap: 4815; seed обучения: 17,29,43. Диагностика использует независимые
диапазоны seed 1200000+, 1400000+ и 1600000+; timing — 1700000+.
Истории используют исходный PCG64 движка, а не отдельную реализацию
случайного транспорта в ML-коде.

Запись mc_records.csv содержит три входа, group_id, run_configuration_id,
seed, target, successes, histories, SE, CI, status, timing и версию алгоритма.
Исходные bundles дополнительно содержат полный материал, кодовые хеши,
параметры, предупреждения и provenance. dataset.csv объединяет только повторы
одной группы; в нём сохраняются pooled successes/histories и replicate SEM.
splits.json хранит имена частей и все group_id.

В mlp.json записаны схема v2, упорядоченная feature_spec, единицы и домены,
scaler, widths, численные theta, optimizer/options, seed, статус и metadata
обучения. Все весовые матрицы и смещения восстанавливаются из theta по widths.
Для полинома сохранён порядок квадратов и взаимодействий, определённый
polynomial_design. Загрузка отклоняет несовместимую схему или нечисловые веса.
Теоретически невозможно обнаружить перестановку столбцов голой числовой
матрицы без имён; CLI формирует строку из именованных аргументов, и этот
контракт документирован.

### 12.2. Воспроизведение и контроль артефактов

Подробные команды и правила возобновления:
[ML_REPRODUCIBILITY.md](ML_REPRODUCIBILITY.md). Режим evaluate читает готовую
модель, заново валидирует native-датасет и оценивает исходный test без fit.
Дополнительный --dataset допускает только новые совместимые конфигурации
и запрещает пересечение с train/validation. Predict требует именованных
толщины, поглощения и рассеяния; за пределами домена применяется отказ.

Каждая из 13 таблиц сгенерирована из JSON/CSV, а не перепечатана вручную.
report_tables.json хранит их машинный вид и источник; report_consistency.json
подтверждает точное совпадение табличных блоков отчёта и comparison.csv
с metrics.json. manifest.json перечисляет артефакты, размеры и SHA256.
Он служит проверке целостности; любой внешний автор может заменить файл
и его checksum, поэтому аутентификация этим механизмом не заявляется.

### 12.3. Перечень результатов

experiment_config.json, pilot.json, splits.json, dataset.csv, mc_records.csv,
experiments.zip, generation.json и generation_progress.json описывают
план и данные. dummy.json, ridge.json, polynomial_ridge.json и mlp.json —
модели. selection.json и training_trials.json сохраняют все попытки;
learning_curves.csv — историю выбранного fit. study_models.json,
ablation.csv и dataset_size_study.csv содержат дополнительные fit.
metrics.json, comparison.csv, test_predictions.csv и error_regions.csv —
основную оценку. diagnostic_experiments.zip, diagnostics.json,
diagnostic_predictions.csv и noise_study.csv — независимую диагностику.
timing.json содержит отдельные повторы времени и способы учёта setup.
figures/ содержит 13 PNG, 300 dpi. manifest.json, report_tables.json,
report_consistency.json, cli_checks.json и test_execution.json фиксируют
проверки по мере их фактического выполнения.

Сырые ZIP и большие checkpoint-модели сохраняются локально и исключены
из автоматического добавления в Git. В публичном репозитории доступны
JSON-модели, сводные данные, метрики и рисунки. Predict использует JSON-модель;
evaluate требует также native-архив experiments.zip.''')
    paragraph("### 12.4. Хеши кода\n\n"+"\n".join(f"- `{name}`: `{digest}`" for name,digest in config["source_hashes"].items()))
    revision_path=directory/"postprocessing_revision.json"
    if revision_path.exists():
        paragraph("Указанные хеши относятся к выполненным расчётам. Первичная "
            "постобработка рисунков и пояснений записана в postprocessing_revision.json. "
            "Последующая редакционная чистка комментариев и документации описана в "
            "ML_REPRODUCIBILITY.md; текущие исходники могут иметь другие хеши. "
            "Веса, данные и численные метрики сохранены.")
    tests_path=directory/"test_execution.json"
    if tests_path.exists():
        paragraph("### 12.5. Фактически выполненная проверка\n\n```json\n"+json.dumps(read_json(tests_path),ensure_ascii=False,indent=2)+"\n```\n\nТесты устанавливают программную корректность в указанном объёме; инженерная валидация из их прохождения не следует.")
    report_path=ROOT/"docs/ML_RESEARCH_REPORT.md"
    report_path.write_text("\n".join(blocks),encoding="utf-8")
    verify_report_v2(directory,report_path)
    manifest_v2(directory,config)
    return report_path


def run_v2(mode,directory,budget_seconds):
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"manifest.json").exists():
        raise FileExistsError("completed v2 results preserved; use evaluate/predict or a new directory")
    started=perf_counter()
    plan=pilot(directory,mode=="quick",version=VERSION_V2,budget_seconds=budget_seconds)
    config,records,grouped=generate(directory,plan)
    arrays=partition_arrays(grouped,config["partitions"],feature_names=config["feature_names"])
    models,selection=train(directory,arrays,config)
    studies=studies_v2(directory,arrays,models["mlp"],selection,config)
    if (directory/"metrics.json").exists():
        results=read_json(directory/"metrics.json")
    else:
        results=evaluate_v2(directory,models,arrays,grouped,selection,studies,config)
        results["checks"]=dict(source_hashes_unchanged=hashes()==config["source_hashes"],group_disjoint=True,
            scaler_train_only=bool(np.array_equal(np.asarray(models["mlp"]["mean"]),arrays["train"][0].mean(axis=0))),
            model_roundtrip=True,repeat_training_difference=selection["reproducibility_max_difference"],
            test_used_for_selection=False,all_runs_complete=all(r["status"]=="completed" for r in records))
        write_json(directory/"metrics.json",results)
        save_csv(directory/"test_predictions.csv",results["test_predictions"])
    diagnostics=diagnostic_v2(directory,models,config)
    timing=benchmark(directory,models,arrays["test"][0],config,selection,plan)
    if not (directory/"comparison.csv").exists():
        save_csv(directory/"comparison.csv",[dict(model=n,n_test=len(arrays["test"][1]),**results[n]["test"],
            training_seconds=m["training_seconds"],predict_seconds=timing["models"][n]["single"]["median_seconds"]) for n,m in models.items()])
    plots_v2(directory,models["mlp"],results,timing,diagnostics)
    atomic_json(directory/"run_completion.json",dict(wall_seconds_current_invocation=perf_counter()-started,
        status="experiments_complete",report="generated separately after verification; no final claims before measurements"))
    manifest_v2(directory,config)
    print(json.dumps(dict(output=str(directory),test=results["mlp"]["test"],speedup=timing["speedup"],
                          deployment_break_even=timing["deployment_break_even"]),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("quick","full","evaluate","predict","report"))
    parser.add_argument("--schema",choices=("v1","v2"),default="v2",help="Generation schema; saved models carry their own schema")
    parser.add_argument("--budget-seconds",type=float,default=300.,help="Maximum time per main-generation invocation (resumable); other finite stages separate")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--dataset",type=Path,help="Optional independent dataset directory for evaluate")
    parser.add_argument("--thickness",type=float,help="cm")
    parser.add_argument("--absorption-rate",type=float,help="synthetic 1/cm, required by v2")
    parser.add_argument("--scattering-rate",type=float,help="synthetic 1/cm, required by v2")
    args=parser.parse_args()
    if args.output is None:
        args.output=ROOT/("benchmarks/results/ml_multivariate_v2" if args.schema=="v2" else "benchmarks/results/ml_baseline")
    if not np.isfinite(args.budget_seconds) or args.budget_seconds<=0:
        parser.error("--budget-seconds must be positive and finite")
    if args.mode=="report":
        print(research_report_v2(args.output))
    elif args.mode in ("quick","full"):
        if args.schema=="v2":
            run_v2(args.mode,args.output,args.budget_seconds)
        else:
            run(args.mode,args.output)
    elif args.mode=="predict":
        if args.thickness is None:
            parser.error("predict requires --thickness")
        model=load_model(args.output/"mlp.json")
        if model["schema"]==VERSION_V2:
            named=dict(thickness_cm=args.thickness,absorption_rate=args.absorption_rate,scattering_rate=args.scattering_rate)
            if any(named[n] is None for n in model["features"]):
                parser.error("v2 predict requires --thickness, --absorption-rate and --scattering-rate")
            x=[[named[n] for n in model["features"]]]
            print(json.dumps(dict(inputs=named,transmission=float(predict(model,x,feature_names=model["features"])[0]),
                                  schema=VERSION_V2,scope="synthetic educational simulator emulation")))
        else:
            if args.absorption_rate is not None or args.scattering_rate is not None:
                parser.error("v1 water model accepts thickness only")
            print(json.dumps(dict(thickness_cm=args.thickness,transmission=float(predict(model,[[args.thickness]])[0]),
                                  profile=REFERENCE_POINT_PROFILE,scope="simulator emulation, not physical validation")))
    else:
        model=load_model(args.output/"mlp.json")
        directory=args.dataset or args.output
        _, grouped=load_dataset(directory)
        config=read_json(args.output/"experiment_config.json")
        if args.dataset:
            prohibited=set(config["partitions"]["train"]+config["partitions"]["validation"])
            if prohibited & {r["configuration_id"] for r in grouped}:
                raise ValueError("evaluation dataset overlaps training/validation groups")
            x=np.array([[r[n] for n in model["features"]] for r in grouped]); y=np.array([r["target"] for r in grouped])
        else:
            x,y=partition_arrays(grouped,config["partitions"],feature_names=model["features"])["test"]
        print(json.dumps(dict(n=len(y),metrics=metrics(y,predict(model,x)),retrained=False),indent=2))


if __name__=="__main__":
    main()
