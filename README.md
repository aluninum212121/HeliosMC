# HeliosMC

HeliosMC is an educational Python simulator for Monte Carlo neutron transport. It models particle histories, material interactions, generation statistics, and a Matplotlib interface for inspecting simulation results. It is intended for learning and software experimentation; it is not a validated reactor-design or safety-analysis tool.

## Run

Install the approved dependencies listed in `requirements.txt` when that file is present, then launch the GUI from the project root:

```powershell
py main.py
```

Numba is optional. When it is unavailable, `numba_compat.py` provides the Python execution fallback.

## Project modules

- `main.py` starts the application.
- `physics.py` handles transport, fission source generation, generation statistics, and the existing multiplication estimate.
- `material.py` defines material properties and the built-in material library.
- `neutron.py` contains the standalone single-particle representation; the main batch transport path uses NumPy arrays.
- `gui.py` presents simulation controls, plots, and completion status.
- `simulation_worker.py` runs simulations outside the GUI event loop.
- `numba_compat.py` enables optional Numba acceleration and the Python fallback.

## Results and limitations

The generation ratio retains its established definition: fission production divided by the number of histories transported in that generation. Histories that reach the step limit remain in that denominator and are reported separately as unfinished. Results with unfinished histories are marked incomplete and should not be interpreted as a settled physical estimate. The code and regression tests do not establish physical validity.

Seeded runs are reproducible within a fixed backend and software environment. Exact random sequences are not guaranteed across Python, NumPy, Numba, platform, or backend changes.

Materials are resolved once per run to structurally checked, immutable `TransportData` in
`material.py`. Existing `Material` objects remain supported; another provider can
implement `get_transport_data()` to return the same constant one-group inputs.
This boundary does not implement energy-dependent tables or ENDF processing.
Results retain their existing fields and also describe the estimator, warnings
and run configuration. See the [current architecture addendum](docs/ARCHITECTURE_AUDIT.md#material-boundary-and-result-contract-2026-10-31)
for units, contracts and the next integration steps.

`MaterialDataset` describes identity, composition, temperature, provenance and
scalar or pointwise `MaterialQuantity` values with explicit units and availability.
`get_builtin_dataset(key)` describes an existing preset as legacy/unverified data,
without guessing its unknown scientific metadata. Pointwise arrays are checked
and stored immutably; the current engine rejects them rather than silently
reducing them. See the [structured material schema](docs/ARCHITECTURE_AUDIT.md#structured-material-data-schema-2026-10-31)
for an example, validation rules and the proposed future ML dataset manifest.

Material metadata now groups state (`MaterialState`), lineage (`DataOrigin`),
review scope (`DataQuality`) and optional source-parameter standard uncertainty
(`ParameterUncertainty`). Unknown values remain `None`. Historical preset numbers
and `TransportData` are preserved. `require_scientific_reference()` warns and
rejects unverified records; a declared review is not an automated scientific proof.
The legacy multiplication kernel keeps scientific datasets storage-only,
including one-group records. The explicit nonreactive reference adapter below
has its own narrower contract.

`load_source_catalog()` reads the offline [official-source registry](docs/data_sources.json).
It performs no downloads. `inspect_local_file()` computes a local SHA-256; integrity
verification additionally requires an expected hash and its independent reference.
`load_synthetic_table()` reads bounded local JSON fixtures with schema
`heliosmc.synthetic-material.v1`, checks units/shapes/missing values/strict energy
ordering and keeps a file-check record. Only explicit synthetic records are
accepted. JSON schema validity, byte integrity and scientific validity are distinct.
See the [provenance schema and source research](docs/ARCHITECTURE_AUDIT.md#material-provenance-schema-and-source-catalog-2026-10-31)
for the file contract, source-access limitations, format comparison and Dataset Pipeline v1 design.

## Verified educational scenarios

The nonreactive `physics.simulate_slab` API models independent random flights
through one or more abstract layers. By default each layer is an explicitly synthetic or
educational `MaterialDataset` with only `absorption_rate` and `scattering_rate`
(both in `1/cm`); spatial interfaces are supplied separately in cm. There are no
nuclear reactions or particle multiplication in this path. Existing GUI behavior
and the experimental generation estimator remain compatible.

SciPy 1.18.1 was installed and tested with Python 3.14.4, NumPy 2.4.4 and
Numba 0.67.0. It supplies uncertainty/interpolation outside JIT; mathematical tests
also use numerical quadrature. New statistics/interpolation functions require
SciPy; legacy application imports remain usable without it. No requirements file
was introduced because the project previously had none.

```python
from material import MaterialDataset, MaterialQuantity
from physics import simulate_slab

medium = MaterialDataset(
    "example:abstract", "Synthetic teaching medium", "synthetic", "invented example",
    "1", ("Abstract rates, not measured nuclear data.",),
    (MaterialQuantity("absorption_rate", "1/cm", .5),
     MaterialQuantity("scattering_rate", "1/cm", 0.)),
)
result = simulate_slab([medium], [0., 2.], n_histories=10000, seed=123)
print(result["estimates"]["escaped_right"])
```

Complete IID histories receive marginal exact binomial intervals. Censored or
empty runs explicitly have unavailable full-outcome estimates/intervals.
`summarize_samples` defaults to descriptive statistics; its Student-t interval
requires the caller's explicit `iid_normal` assumption. It is not an uncertainty
method for correlated generation ratios. Synthetic-only PCHIP interpolation is
available separately and rejects extrapolation; no transport reads those tables.
See the [mathematical audit](docs/ARCHITECTURE_AUDIT.md#educational-mathematics-upgrade-2026-10-31)
for derivations, numerical limits, reproducibility, performance and test evidence.

## Reference data and research export (2026-11-06)

`get_reference_dataset("Вода (H₂O)")` and `get_reference_dataset("Графит")`
return separate records in the existing material schema. NIST supplies chemical
identity and conventional molar mass. The water density is the IAPWS published
correlation check value at **298.15 K, 100000 Pa**. Graphite density remains
missing because grade/porosity/state are unknown. `source_checked` means the
quoted property and its scope were checked; it does not mean the transport
model or a real specimen was scientifically validated. All four historical
`MATERIALS` presets retain every number and are `legacy_unverified`.

The v2 schema adds pressure metadata, a separate source verification status,
`not_applicable` for energy-independent reference properties, and an optional
`AveragingConditions` contract. Unknown averaging cannot establish a validated
one-group nuclear parameter. These thermophysical-only records remain separate
from executable point-data records described below.

```python
from material import get_reference_dataset, export_material_reference, load_material_reference
from physics import (create_slab_configuration, run_slab_experiment,
                     export_slab_experiment, load_slab_experiment)

# Use the synthetic `medium` from the example above. Choose criteria BEFORE running.
config = create_slab_configuration("absorption-demo", [medium], [0., 2.],
    n_histories=4000, seed=123, verification="pure_absorption_v1")
experiment = run_slab_experiment(config)
digest = export_slab_experiment(experiment, "my-new-experiment")
restored = load_slab_experiment("my-new-experiment", expected_sha256=digest)
replay = run_slab_experiment(restored["configuration"])
assert replay["result"]["outcome_counts"] == restored["result"]["outcome_counts"]

water = get_reference_dataset("Вода (H₂O)")
water_digest = export_material_reference(water, "my-new-water.json")
assert load_material_reference("my-new-water.json", expected_sha256=water_digest).state.pressure == 100000.
```

The destination must be new. Each experiment has `experiment.json`, `results.csv`
and `arrays.npz`. Load checks versions, hashes, numeric shapes/dtypes, counts,
uncertainty, verification and feature/target separation. NPZ loads disable pickle;
no trajectory arrays are exported. Retain returned hashes independently: a
self-contained checksum detects corruption, not an author who replaces all files.
Replay requires matching code hashes and recorded dependency versions.

`split_slab_groups(experiments)` keeps repeats of the same mathematical problem
in one partition, including runs with different seeds/populations. Only complete,
analytically checked synthetic/educational experiments qualify; a small collection
may give an empty partition. No ML model is trained. The [research audit](docs/ARCHITECTURE_AUDIT.md#scientific-material-data-and-research-export-2026-11-06)
contains field/source tables, criteria, export contracts and current test evidence.
The audit records one executed small example, its observed uncertainty, and
the reproducible export/reload contract. Its 95% interval missed the analytic
value in that run; the fixed alpha=1e-6 regression criterion passed. No seed
selection or after-the-fact tolerance adjustment was used.

## Explicit reference transport (2026-11-10)

The nonreactive slab now accepts `material_mode="reference"`. Its reviewed
adapter uses **JENDL-5 numerical point tables at 0.0253 eV, processed at 300 K**,
representative NIST isotope proportions and `Sigma=sum(N_i*sigma_i)` in cm^-1.
These are point values, with no broad-spectrum averaging. The user must select
the `free_gas_point_isotropic_v1` approximation: the kernel still freezes energy
and scatters isotropically in the laboratory. Real water/graphite thermal
scattering requires bound-state data; this mode does not predict moderation.

```python
from material import get_reference_transport_dataset, REFERENCE_APPROXIMATION
from physics import (create_slab_configuration, run_slab_experiment,
                     export_slab_experiment, load_slab_experiment)

water = get_reference_transport_dataset("Вода (H₂O)",
    approximation=REFERENCE_APPROXIMATION)  # 300 K, 100000 Pa, IAPWS density
config = create_slab_configuration("water-point-example", [water], [0., .2],
    n_histories=1000, seed=701, material_mode="reference", backend="python")
experiment = run_slab_experiment(config)
digest = export_slab_experiment(experiment, "water-point-example")  # new directory
restored = load_slab_experiment("water-point-example", expected_sha256=digest)
assert restored == experiment  # parameters, sources, seed, SE and CI preserved
replay = run_slab_experiment(restored["configuration"])
assert replay["result"]["outcome_counts"] == restored["result"]["outcome_counts"]
```

Graphite requires explicit `density` in g/cm^3 and a `density_origin=DataOrigin(...)`
describing sample/grade, temperature and acquisition method. This input retains
unverified status. No density or uranium reference configuration is guessed.
`get_reference_dataset()` retains the earlier thermophysical-only records.
Mixing reference and explicit-rate layers raises an error. Calls without the
new keyword preserve their original behavior; historical `MATERIALS` and
multiplication APIs retain their data and algorithms.

New configurations use v2 with material mode/model version; v1 archives remain
readable through the same JSON/CSV/NPZ reader. Code/environment hashes still
guard exact replay. Reference runs carry explicit approximation warnings and
are excluded from the current mathematically certified training subset.
Their source checks do not establish real-material scientific validation.
The [current audit](docs/ARCHITECTURE_AUDIT.md#reference-material-model-2026-11-10)
contains every used isotope value, its source, assumptions and test evidence.

## Tests

Run the automated tests from the project root:

```powershell
py -m pytest -q
```

## Benchmarks and documentation

Benchmark scripts are in `benchmarks/`; run them as modules from the project root, for example:

```powershell
py -m benchmarks.benchmark_transport --variant current --output benchmarks/results/current_numba.json
py -m benchmarks.benchmark_generation --mode performance --large
py -m benchmarks.benchmark_transport --variant current --educational --baseline-source <saved-before-physics.py> --output benchmarks/results/educational_math_numba.json
```

Recorded benchmark outputs are in `benchmarks/results/`. Architecture, physics notes, audit records, and performance reports are in `docs/`. See [the performance upgrade report](docs/PERFORMANCE_UPGRADE_REPORT.md) for measured comparisons and limitations.
