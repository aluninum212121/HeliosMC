# HeliosMC

HeliosMC is an educational Python simulator for Monte Carlo neutron transport.
It combines a desktop interface, reproducible experiment exports, and a study
of machine learning surrogate models.

## Features

- Simulate independent neutron histories through one or more layers, including
  absorption, scattering, and escape through either boundary.
- Run with explicitly specified educational material rates or select documented
  water and graphite reference profiles with their sources and approximations.
- Explore an experimental generation scenario with fission and source
  reproduction.
- Inspect outcome charts, parameter and uncertainty tables, sampled trajectories,
  and a map of registered fission events.
- Export completed experiments as JSON/CSV/NPZ archives, then load and validate
  those archives.
- Compare MLP, constant, linear Ridge, and polynomial Ridge models against
  results from the synthetic Monte Carlo simulator.

HeliosMC is intended for education and software research. It has not been
validated for engineering calculations of real systems, criticality, or
radiation safety.

## Installation and launch

Install Python with Tkinter. The recorded research run used Python 3.14.4;
versions of the numerical libraries are listed in the
[reproducibility guide](docs/ML_REPRODUCIBILITY.md).

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install numpy scipy matplotlib
.venv\Scripts\python main.py
```

On Linux or macOS, use `.venv/bin/python`. Tkinter may need to be installed
as a system package. On Windows, use your installed interpreter or the `py`
launcher if the `python` command is unavailable.

NumPy, SciPy, and Matplotlib are the core dependencies. Numba is optional; the
application falls back to Python/NumPy when Numba is unavailable. To enable
acceleration and run the test suite:

```powershell
.venv\Scripts\python -m pip install numba pytest
```

Numba accelerates supported calculations; the first call includes JIT
compilation. The desktop interface uses the standard Tkinter/ttk widgets and
Matplotlib TkAgg backend.

## Using the interface

1. Choose a scenario and material or educational medium, then enter its
   parameters.
2. Set a random seed if you need reproducible results within a fixed software
   environment.
3. Run the simulation and inspect the Overview, Visualization, and Data tabs.
4. Export a completed experiment to a new directory or save a plot.

Transport through a slab is the startup scenario. Reference graphite requires
an explicitly supplied density, sample description, and measurement method.
Point values at 0.0253 eV are identified separately from averages over a
specified spectrum. The Visualization tab is hidden when spatial data are not
available.

The generation scenario reports a source reproduction ratio, the values for
individual generations, and their descriptive spread. This ratio is not a
validated physical `k_eff`.

Simulation runs in a background thread. Cancellation is handled between
generations or history batches. If the form changes, displayed results are
marked as belonging to the previous run; exports retain that run's parameters.
The Open command reads an `experiment.json` file together with its CSV/NPZ
sidecars. A loaded configuration is preserved for replay; use New to configure
another experiment.

## Run a simulation without the GUI

```python
from material import MaterialDataset, MaterialQuantity
from physics import simulate_slab

medium = MaterialDataset(
    "example:abstract", "Educational medium", "synthetic", "Mathematical example",
    "1", ("Specified rates; not measured nuclear data.",),
    (MaterialQuantity("absorption_rate", "1/cm", 0.5),
     MaterialQuantity("scattering_rate", "1/cm", 0.0)),
)
result = simulate_slab([medium], [0.0, 2.0], n_histories=10000, seed=123)
print(result["outcome_counts"])
print(result["estimates"]["escaped_right"])
```

Coordinates and thicknesses are in cm; interaction rates are in cm⁻¹. The
[architecture and model guide](docs/ARCHITECTURE_AUDIT.md) describes the
mathematical checks, API contracts, material profiles, and data provenance.

## Machine learning study

The saved v2 study emulates the fraction of histories that escape through the
right boundary of a synthetic slab using three inputs:

| Input | Range |
| --- | --- |
| Thickness | 0.1–3 cm |
| Absorption rate | 0.02–1.2 cm⁻¹ |
| Scattering rate | 0–2 cm⁻¹ |

The full experiment contains 900 configurations, each with three Monte Carlo
replicates of 8,000 histories. Data are split by configuration so that
replicates from one configuration cannot appear in both training and test
sets. The dataset, JSON models, metrics, and figures are in
[`benchmarks/results/ml_multivariate_v2/full/`](benchmarks/results/ml_multivariate_v2/full/).

Predict with the saved model without retraining:

```powershell
python -B -m benchmarks.ml_baseline predict --output benchmarks/results/ml_multivariate_v2/full --thickness 1.5 --absorption-rate 0.4 --scattering-rate 1.0
```

Run a new quick study in a new output directory:

```powershell
python -B -m benchmarks.ml_baseline quick --schema v2 --output benchmarks/results/ml_multivariate_v2/quick_new --budget-seconds 300
```

The time budget applies to one main data-generation call; training and
diagnostics have separate execution limits. The
[reproducibility guide](docs/ML_REPRODUCIBILITY.md) contains commands for the
full study, evaluation, and resuming interrupted runs. Methodology and
measured results are in [ML_RESEARCH_REPORT.md](docs/ML_RESEARCH_REPORT.md).

Compact results and full JSON models are included in the repository. Raw
experiment ZIP archives and large checkpoints remain local and are excluded
from Git. The `predict` command uses the JSON model; `evaluate` also requires
`experiments.zip`. Generate a new dataset in a separate output directory using
the reproducibility guide.

## Limitations and reproducibility

- Slab transport uses constant rates, absorption, and isotropic scattering
  without energy changes.
- Reference profiles retain their sources and conditions, but the approximation
  does not model slowing down, thermalization, or bound-state effects in water
  or graphite.
- Unfinished histories are reported explicitly. They are not complete outcomes,
  so full statistical estimates are unavailable for censored runs.
- Binomial intervals describe Monte Carlo sampling uncertainty, not uncertainty
  in material parameters or the physical model. The spread of generation ratios
  is not a standard error for independent observations.
- A seed is reproducible with a fixed backend, code, and environment. Matching
  random sequences across library versions, platforms, and backends is not
  guaranteed.
- Archives store source-code hashes. Even a comment edit changes a hash. Old
  results remain readable, but exact replay may require the original source
  snapshot. Historical hashes and measured results are preserved.

## Tests

Run the test suite from the project root:

```powershell
python -B -m pytest -q -p no:cacheprovider
```

Most GUI tests run headlessly with Agg. The native Tk test requires an
interactive desktop and can be enabled separately:

```powershell
$env:HELIOSMC_NATIVE_GUI_TESTS = "1"
python -B -m pytest tests/test_physics.py::test_gui_native_material_bindings_tab_restore_seed_viewer_and_resize -q -p no:cacheprovider
```

Performance measurements and their limitations are documented in the
[performance report](docs/PERFORMANCE_UPGRADE_REPORT.md). The frozen
`baseline_physics.txt` is required by regression tests; `secondary_stage_physics.txt`
is used for the historical event-storage comparison. Neither file is imported
by the application.

## Project structure

| Path | Purpose |
| --- | --- |
| `main.py` | Application entry point |
| `gui.py` | Interface, plots, experiment loading, and export |
| `physics.py` | Transport, generations, statistics, and experiment archives |
| `material.py` | Materials, data schemas, reference profiles, and sources |
| `neutron.py` | Standalone educational particle model |
| `numba_compat.py` | Numba acceleration and Python fallback |
| `simulation_worker.py` | Background execution and cooperative cancellation |
| `tests/` | Regression, mathematical, and integration tests |
| `benchmarks/` | Performance measurements and ML study |
| `benchmarks/results/` | Saved measurements and research results |
| `docs/` | Model, data provenance, reports, and documentation |
