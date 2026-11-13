"""CPU-only surrogate study of the unchanged reference-water slab simulator.

Run as ``python -m benchmarks.ml_baseline quick|full|evaluate|predict``.
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

import physics
from material import (REFERENCE_APPROXIMATION, REFERENCE_POINT_PROFILE,
                      get_reference_transport_dataset, material_to_dict)

VERSION = "water_slab_surrogate_v1"
ROOT = Path(__file__).resolve().parents[1]
DOMAIN = (0.05, 3.0)  # cm; fixed normal incidence, source at the left face
FEATURES = ["thickness_cm"]
TARGET = "escaped_right_fraction"
SEEDS = (17, 29, 43)  # fixed before observing validation or test targets
CANDIDATES = (((8,), 1e-6), ((16,), 1e-6), ((16, 8), 1e-6), ((16,), 1e-4))


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


def inputs(values, domain=DOMAIN):
    """Require an explicit numeric (n, 1) matrix; never silently add features."""
    raw = np.asanyarray(values)
    if np.ma.isMaskedArray(raw) or raw.dtype.kind not in "fiu":
        raise ValueError("features must be real numeric values without masks")
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[1] != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("features must have finite shape (n, 1)")
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
    actual = targets(actual, len(actual))
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError("invalid prediction vector")
    error = predicted - actual
    ss = float(np.sum((actual - actual.mean())**2))
    return dict(mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
                r2=float(1 - np.sum(error**2)/ss) if ss > 0 else None,
                max_absolute_error=float(np.max(abs(error))), bias=float(np.mean(error)),
                outside_fraction=int(np.count_nonzero((predicted < 0) | (predicted > 1))))


def unpack(theta, widths):
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
              validation=None, maxiter=2000):
    """Scaler and estimator are one artifact; fit sees training data only."""
    x = inputs(x)
    y = targets(y, len(x))
    start = perf_counter()
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    z = (x - mean) / scale
    preprocessing = perf_counter() - start
    model = dict(schema=VERSION, kind=kind, features=FEATURES, target=TARGET,
                 domain=list(DOMAIN), mean=mean.tolist(), scale=scale.tolist(),
                 alpha=alpha, seed=seed, preprocessing_seconds=preprocessing,
                 train_min=float(x.min()), train_max=float(x.max()))
    start = perf_counter()
    if kind == "dummy":
        model.update(value=float(y.mean()), success=True, iterations=0)
    elif kind == "ridge":
        design = np.column_stack((z, np.ones(len(z))))
        penalty = np.diag([alpha, 0.])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        model.update(coefficients=coefficients.tolist(), success=True, iterations=1)
    elif kind == "mlp":
        widths = [1, *hidden, 1]
        rng = np.random.default_rng(seed)
        theta = np.concatenate([part.ravel() for left, right in zip(widths[:-1], widths[1:])
            for part in (rng.uniform(-np.sqrt(6/(left+right)), np.sqrt(6/(left+right)), (left, right)),
                         np.zeros(right))])
        history = []
        if validation is not None:
            vx = (inputs(validation[0]) - mean) / scale
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


def predict(model, x):
    x = inputs(x, model["domain"])
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    if model["kind"] == "dummy":
        return np.full(len(x), model["value"])
    if model["kind"] == "ridge":
        return np.column_stack((z, np.ones(len(z)))) @ np.asarray(model["coefficients"])
    if model["kind"] == "mlp":
        return forward(np.asarray(model["theta"]), model["widths"], z)[0]
    raise ValueError("unknown estimator")


def load_model(path):
    """JSON numeric parameters only: no pickle or executable deserialization."""
    if Path(path).stat().st_size > 10_000_000:
        raise ValueError("model file too large")
    model = read_json(path)
    if (model.get("schema") != VERSION or model.get("features") != FEATURES
            or model.get("target") != TARGET or model.get("domain") != list(DOMAIN)
            or model.get("kind") not in ("dummy", "ridge", "mlp")):
        raise ValueError("unsupported model schema or feature contract")
    for name in ("mean", "scale"):
        a = np.asarray(model[name], dtype=float)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError("invalid scaler")
    if model["scale"][0] <= 0 or not model.get("success"):
        raise ValueError("invalid scaler or unconverged model")
    if model["kind"] == "mlp":
        widths = model["widths"]
        if (len(widths) not in (3, 4) or widths[0] != 1 or widths[-1] != 1
                or any(type(n) is not int or not 1 <= n <= 64 for n in widths)):
            raise ValueError("unsupported architecture")
    if not np.isfinite(predict(model, [[sum(DOMAIN)/2]])).all():
        raise ValueError("nonfinite model")
    return model


def configuration(thickness, seed, histories):
    return physics.create_slab_configuration("ml-reference-water", [water()], [0., float(thickness)],
        histories, seed=int(seed), max_collisions=1000, max_samples=0, source_mu=1.,
        source_x=0., material_mode="reference", backend="auto")


def observation(experiment):
    """Explicit emulation QC; never changes the archive's certification flag."""
    c, r = experiment["configuration"], experiment["result"]
    p = c["parameters"]
    if (c["material_mode"] != "reference" or c["reference_profile"]["profile_id"] != REFERENCE_POINT_PROFILE
            or c["layers"] != [material_to_dict(water())] or p["source_x"] != 0.
            or p["source_mu"] != 1. or len(p["edges_cm"]) != 2 or p["edges_cm"][0] != 0.):
        raise ValueError("incompatible material, profile, geometry or source")
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
                status=r["execution_status"], certified_training_eligible=
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
        if len({r["thickness_cm"] for r in rows}) != 1:
            raise ValueError("group feature mismatch")
        n = sum(r["histories"] for r in rows)
        k = sum(r["successes"] for r in rows)
        stats = physics.summarize_binomial(k, n)
        # Pooled Bernoulli SE and independent-replicate SEM are distinct diagnostics.
        replicate_sem = float(np.std([r["target"] for r in rows], ddof=1)/np.sqrt(len(rows))) if len(rows)>1 else None
        result.append(dict(configuration_id=group, thickness_cm=rows[0]["thickness_cm"],
            target=stats["estimate"], standard_error=stats["standard_error"],
            ci_low=stats["confidence_interval"][0], ci_high=stats["confidence_interval"][1],
            replicate_sem=replicate_sem, repeats=len(rows), histories=n))
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


def pilot(directory, quick):
    """A real small experiment precedes selection of the main compute budget."""
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


def partition_arrays(grouped, partitions):
    by_group = {r["configuration_id"]: r for r in grouped}
    flattened = [g for ids in partitions.values() for g in ids]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(by_group):
        raise ValueError("split overlap or missing groups")
    return {name: (np.array([[by_group[g]["thickness_cm"]] for g in ids]),
                   np.array([by_group[g]["target"] for g in ids])) for name, ids in partitions.items()}


def train(directory, arrays, config):
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("quick","full","evaluate","predict"))
    parser.add_argument("--output",type=Path,default=ROOT/"benchmarks/results/ml_baseline")
    parser.add_argument("--dataset",type=Path,help="Optional independent dataset directory for evaluate")
    parser.add_argument("--thickness",type=float,help="cm; only the saved water profile is supported")
    args=parser.parse_args()
    if args.mode in ("quick","full"):
        run(args.mode,args.output)
    elif args.mode=="predict":
        if args.thickness is None:
            parser.error("predict requires --thickness")
        model=load_model(args.output/"mlp.json")
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
            x=np.array([[r["thickness_cm"]] for r in grouped]); y=np.array([r["target"] for r in grouped])
        else:
            x,y=partition_arrays(grouped,config["partitions"])["test"]
        print(json.dumps(dict(n=len(y),metrics=metrics(y,predict(model,x)),retrained=False),indent=2))


if __name__=="__main__":
    main()
