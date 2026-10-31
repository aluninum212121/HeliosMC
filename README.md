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
```

Recorded benchmark outputs are in `benchmarks/results/`. Architecture, physics notes, audit records, and performance reports are in `docs/`. See [the performance upgrade report](docs/PERFORMANCE_UPGRADE_REPORT.md) for measured comparisons and limitations.
