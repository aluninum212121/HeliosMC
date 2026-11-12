# HeliosMC

HeliosMC is an educational Python simulator for Monte Carlo neutron transport. It models particle histories, material interactions, generation statistics, and a Matplotlib interface for inspecting simulation results. It is intended for learning and software experimentation; it is not a validated reactor-design or safety-analysis tool.

## Run

Install the approved dependencies listed in `requirements.txt` when that file is present, then launch the GUI from the project root:

```powershell
py main.py
```

Numba is optional. When it is unavailable, `numba_compat.py` provides the Python execution fallback.

## Experiment workspace

The desktop interface uses standard Tkinter/ttk with embedded Matplotlib
(`TkAgg`). Its light workspace has a top action bar, a scrollable parameter
sidebar, result tabs and a bottom status/progress bar. No additional GUI package
is required. The normal workflow is **choose a scenario → enter parameters →
run → inspect results → export**.

- **Generations · legacy** preserves every historical material preset and the
  existing source-generation algorithm. The overview distinguishes the selected
  mean from individual generation ratios and from their descriptive spread.
  The ratio is not labelled a validated physical `k_eff`.
- **Transport through a slab** runs the existing independent-history model.
  Choose a documented reference profile or an educational medium with explicitly
  entered absorption/scattering rates. Reference water/graphite choices follow
  the available profiles; graphite additionally requires its density, sample
  description and density method. A point at 0.0253 eV is identified as a point,
  while the existing Maxwell-flux profile is identified as an average.

`Overview` presents compact results; `Graphs` selects one plot at a time;
`Heatmap` shows registered fission coordinates projected onto X–Y with a colorbar;
`Trajectories` shows the retained sample for a selected generation, with optional
playback; `Data` lists the actual inputs, results, seed and provenance. The heatmap
counts fissions rather than neutron flux. Slab research results contain aggregate
outcomes and intentionally omit spatial paths, so the corresponding spatial tabs
explain that no such data are available.

`Open` reads an existing slab experiment's `experiment.json` and checks its
JSON/CSV/NPZ bundle through the existing loader. The loaded configuration is kept
intact for replay; choose `New` to configure a different experiment. `Export`
saves a completed slab experiment into a **new directory**, preserving its seed,
inputs and statistical uncertainty. It also saves the selected plot as PNG at
300 dpi. Historical generation results support plot export; a new research
archive format for them has not been introduced. Standalone, unrun configurations
are not saved as if they were completed experiments.

Simulation work stays in the existing background worker. Stop requests are
cooperative: generation runs stop at generation boundaries and slab runs between
batches. No partial result is presented as completed. Slab progress is
indeterminate because its existing engine has no progress callback. Animation
does not start automatically. Replay retains the existing checks for matching
core code, numerical backend and software environment.

For implementation contracts and verification notes, see the
[GUI workspace audit](docs/ARCHITECTURE_AUDIT.md#experiment-workspace-redesign-2026-11-12).

## Project modules

A [native workspace screenshot](docs/gui_workspace.png) shows the educational
slab overview. Manual desktop interaction was stopped with Escape during this
session; full Windows scaling coverage is not claimed.

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

This initial integration used v2 with material mode/model version; v1 archives remain
readable through the same JSON/CSV/NPZ reader. Code/environment hashes still
guard exact replay. Reference runs carry explicit approximation warnings and
are excluded from the current mathematically certified training subset.
Their source checks do not establish real-material scientific validation.
The [current audit](docs/ARCHITECTURE_AUDIT.md#reference-material-model-2026-11-10)
contains every used isotope value, its source, assumptions and test evidence.

## Documented energy profiles (2026-11-10)

`get_reference_profiles()` lists profile IDs, availability by material, spectrum,
energy bounds, material/processing and neutron-spectrum temperatures, sources,
verification status and limitations. Choose `profile_id=` when constructing a
reference material; omitting it preserves the complete original JENDL-5 point
record. Every layer of an experiment must use the same profile/library.

| Profile ID | Energy definition | Available materials |
| --- | --- | --- |
| `jendl5_point_0p0253ev` | Point at 0.0253 eV | Water and graphite |
| `jendl40_point_0p0253ev` | Independent JENDL-4.0 natural-carbon point control at 0.0253 eV | Graphite |
| `jendl40_maxwell_flux_0p0253ev` | Normalized flux weight E exp(-E/kT), kT=0.0253 eV, bounds 1e-5..10 eV | Graphite |

The [JAEA C-0 table](https://wwwndc.jaea.go.jp/cgi-bin/Tab80WWW.cgi?iso=C000&lib=J40)
publishes MT2/MT102 MACS with a 2/sqrt(pi) prefactor. The group preset removes
that prefactor: sigma_group = sqrt(pi)/2 * MACS. Its neutron-spectrum temperature
is about 293.5943 K; the evaluated material/processing temperature remains 300 K.
The [official report abstract](https://jopss.jaea.go.jp/search/servlet/search?5026620=&language=1)
specifies kT. These JENDL-4.0 records use the elemental C-0 evaluation; no JENDL-5
isotope channels are substituted. Point controls are not spectral averages.

```python
from material import (get_reference_transport_dataset, get_reference_profiles,
                      REFERENCE_MAXWELL_PROFILE, REFERENCE_GROUP_APPROXIMATION)

profiles = get_reference_profiles()
# Supply independently identified sample_density_g_cm3 and sample_origin.
graphite = get_reference_transport_dataset("Графит",
    profile_id=REFERENCE_MAXWELL_PROFILE,
    approximation=REFERENCE_GROUP_APPROXIMATION,
    density=sample_density_g_cm3, density_origin=sample_origin)
# Pass [graphite] with material_mode="reference" to the same slab/export APIs.
```

The group closure holds the imposed spectrum fixed after collisions; it does
not model moderation or spectral hardening. Exponential attenuation using an
averaged coefficient is not the integral of energy-dependent attenuation.
Graphite density remains a required unverified caller input. New water profiles
are explicitly unavailable because a complete source-checked channel set was
not obtained. Fast and epithermal presets are not enabled from incomplete data.

Configuration v3 archives the full profile in configuration, run information
and ML metadata. The same JSON/CSV/NPZ APIs still read v1/v2 archives; seed and
uncertainty remain intact. Reference runs retain their strict training exclusion.
The [profile audit](docs/ARCHITECTURE_AUDIT.md#documented-reference-energy-profiles-2026-11-10)
records the assessment made before implementation and test evidence.

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
