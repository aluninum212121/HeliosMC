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

Materials are resolved once per run to validated, immutable `TransportData` in
`material.py`. Existing `Material` objects remain supported; another provider can
implement `get_transport_data()` to return the same constant one-group inputs.
This boundary does not implement energy-dependent tables or ENDF processing.
Results retain their existing fields and also describe the estimator, warnings
and run configuration. See the [current architecture addendum](docs/ARCHITECTURE_AUDIT.md#material-boundary-and-result-contract-2026-10-31)
for units, contracts and the next integration steps.

`MaterialDataset` describes identity, composition, temperature, provenance and
scalar or pointwise `MaterialQuantity` values with explicit units and availability.
`get_builtin_dataset(key)` describes an existing preset as educational data,
without guessing its unknown scientific metadata. Pointwise arrays are checked
and stored immutably; the current engine rejects them rather than silently
reducing them. See the [structured material schema](docs/ARCHITECTURE_AUDIT.md#structured-material-data-schema-2026-10-31)
for an example, validation rules and the proposed future ML dataset manifest.

## Verified educational scenarios

The nonreactive `physics.simulate_slab` API models independent random flights
through one or more abstract layers. Each layer is an explicitly synthetic or
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
