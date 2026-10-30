# HeliosMC / NeutronMC debug report v0.2

Date: 2026-10-29 (Asia/Tashkent)

## Scope and confirmed findings

The modular architecture and official `main.py` entry point are preserved. The
legacy `NSPM MCNP 3000.py` file was neither edited nor imported. Material constants,
interaction probabilities, distance law, scattering equations, secondary-source
rules, population control, and k calculation were preserved. These checks do not
establish scientific accuracy or reactor validity.

1. **Excessive trajectory storage confirmed.** Transport created a nested list for
   every particle, copied every position, then generation orchestration selected
   only 40 histories. Most allocation was discarded only after transport returned.
2. **Step-limit issue partially fixed already.** The previous stage already
   reported fate -1 in `unfinished` and counted it in the histogram. It was not an
   undefined aggregate outcome. The internal active mask still remained true at
   the limit, and there was no named terminal status.
3. **GUI blocking confirmed.** Run called the full simulation from the event
   callback. Drawing/flushing before the call could not keep the event loop free.
4. **Validation gaps confirmed.** Bad dimensions, nonfinite inputs, zero cross
   sections, zero direction vectors, empty navigation, and invalid configuration
   values could cause failures or invalid states. No worker failure path existed.
5. **Reproducibility gap confirmed.** Global NumPy source generation and Numba's
   separate RNG were not coordinated. Visualization selection also consumed RNG
   draws that could affect subsequent generations.
6. **Extra rendering allocation confirmed.** Heatmap drawing expanded all fission
   arrays into Python row objects and then copied them back to an array.

## Fixes implemented

- Preselect evenly spaced source indices without RNG draws. Transport allocates
  trajectory scratch storage only for the configured sample. Zero samples disables
  path storage. Returned paths own trimmed arrays; no view retains unused scratch.
- Preserve all aggregate accounting independently of paths. Fission coordinates
  remain available for the existing heatmap in both execution modes.
- Remove per-step movement temporaries and scattering vector allocations in the
  hot transport loop. Remove redundant source-array copies. Keep the existing RNG
  draw schedule for transport, including draws for inactive indices.
- Define `HistoryStatus.MAX_STEPS_REACHED = -1`; explicitly terminate remaining
  active histories at the iteration limit. Preserve the compatible `unfinished`
  dictionary key and neutral GUI label. Counts reconcile with processed histories.
- Add one non-daemon background worker with a bounded result queue, cooperative
  cancellation event and explicit IDLE/RUNNING/COMPLETED/FAILED/CANCELLED states.
  Worker computation never calls Matplotlib. A 100 ms canvas timer polls results
  and renders on the main thread. Prevent overlapping jobs and restore controls
  after success, cancellation, computation failure or rendering failure.
- Add Cancel, stop animation when launching work, and request cancellation on
  window close. Release queued results when consumed. Exception messages contain
  their type and text, avoiding retained exception tracebacks and local arrays.
- Validate entry configuration, particle arrays, sampling helpers, material derived
  values and standalone Neutron operations. Support empty results safely and clamp
  display indices. Snapshot submitted configuration for subsequent display.
- Use a local NumPy Generator for sources/population control and explicit per-
  generation seeds for transport. Seeded fallback transport restores its NumPy
  global RNG state. The lock serializes transport calls within this application.
- Accumulate heatmap bins per generation without concatenating all points.
- Fix the RadioButtons activation overload discovered in GUI regression testing:
  use the base Widget activation method to disable it without selecting an option.

## Files modified or added

- `physics.py`: validated public interfaces, sampled transport, explicit terminal
  status, seed handling, cooperative generation checkpoints, reduced copying.
- `material.py`: validation of inputs and derived totals/mean free path.
- `neutron.py`: standalone particle validation.
- `gui.py`: background execution, polling, controls, cancellation, safe rendering,
  bounded heatmap construction, configuration snapshot, `show=False` test support.
- `simulation_worker.py`: worker lifecycle and result communication.
- `tests/test_stability.py`: pytest stability, memory, reproducibility and GUI tests.
- `benchmarks/benchmark_generation.py`: repeatable before/after synthetic benchmark from a Git revision.
- `benchmarks/results/benchmark_v02_results.json`, `benchmarks/results/benchmark_v02_large.json`: raw measurements.
- `docs/verification_gui_v02.png`: rendered existing GUI panels, visually inspected.
- `README.md`: usage, modes, compatibility changes and reproducibility limits.
- `DEBUG_REPORT_v0.2.md`: this report.

`main.py`, `numba_compat.py`, the legacy file and the previous regression tests are
unchanged. A staged `docs/verification_gui.png` already existed before this work and was
left untouched. Temporary implementation scripts and the working baseline copy
were removed. No dependencies were added.

## Tests and validation

Environment: Windows x64, Python 3.14.4, NumPy 2.4.4, Matplotlib 3.10.9,
Numba 0.67.0, pytest 9.1.1.

Commands:

```text
python -m compileall -q main.py gui.py physics.py material.py neutron.py numba_compat.py simulation_worker.py tests/test_physics.py tests/test_stability.py benchmarks/benchmark_generation.py
python -m pytest -q
python -m benchmarks.benchmark_generation --revision 8f7ab377f96c14af70188fcf5c3a20b102fcbdc4
python -m benchmarks.benchmark_generation --mode performance --large
```

The full pytest suite passes **36 tests and 2 subtests**, with no outstanding test
failures. Coverage includes empty arrays and results, bad configuration and shapes,
NaN/infinity, zero direction, iteration limits (0/1/12 steps), dtypes, owning path
arrays, sample limits, disabled paths, input immutability, complete accounting,
same/different seeds, path reproducibility, aggregate/fission coordinate invariance
under sampling, optional Numba fallback in a fresh process, bounded repeated-run
memory, state transitions, worker exceptions, launch/render failures, overlap
prevention, cancellation, close-time worker exit, and successful GUI rendering.

An intermediate GUI test failed because RadioButtons overrides `set_active` for
selection; that regression was corrected and the test now passes. Initial syntax
and test failures during implementation were corrected before final validation.

Final full-suite run: 36 passed and 2 subtests passed in 25.09 seconds. Syntax checks and Git diff whitespace checks passed. The real `python main.py` startup also exited successfully with Agg; its expected non-interactive-backend warning means this was a headless startup check, not a desktop interaction test.

All modular imports work. The `main.main()` entry point is exercised with GUI
construction mocked; the real GUI constructor and rendering are exercised using
Agg. The saved image was inspected: trajectories, k chart, heatmap, balance chart,
status panel and all existing controls remain present. Main-thread polling and
background execution are exercised in tests. Native desktop event-loop latency,
mouse interaction and live animation were not manually measured.

## Benchmark results

Baseline revision: `8f7ab377f96c14af70188fcf5c3a20b102fcbdc4`.
Identical synthetic conditions: 4,000 particles at origin, +z unit directions,
60 steps, mean free path 1, scatter-only cross sections (0,0,1), nu=0,
radius=1e9, seed=2026. This stresses software path storage; it is not a reactor
configuration study. Every particle reaches the explicit step limit.
Each mode runs in a fresh process; JIT warmup is excluded from elapsed time.

| Mode | Time (s) | Peak tracked MiB | Retained tracked MiB | Process peak working set MiB | Recorded paths |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 5.4564 | 81.9679 | 65.5987 | 333.00 | 4,000 |
| Visualization | 0.0442 | 0.5362 | 0.2619 | 160.72 | 40 |
| Performance | 0.0550 | 0.4801 | 0.2009 | 159.98 | 0 |

Tracked peak memory fell by approximately 99.35% with 40 paths and 99.41% with
recording disabled for this workload. All three modes had identical outcome counts
(4,000 unfinished, zero fission/absorption/leakage) and identical final-position
coordinate sums: (246.62371365322065, 200.9941576708254, 3549.425309496216).
The benchmark asserts count equality and numerical coordinate-sum agreement.
Separate mixed-outcome seeded tests check multi-generation sampling invariance.

Larger after-only performance stress run: 100,000 histories x 60 steps, no paths,
0.9614 seconds, 11.9242 MiB tracked peak, 4.9616 MiB tracked retained, 174.97 MiB
process peak working set. All 100,000 histories were explicitly step-limited.

`tracemalloc` measures tracked Python/NumPy allocations, not every native allocator.
Windows process peak working set includes interpreter/JIT startup and warmup, so
it is not simulation-only memory. Timings are single instrumented observations,
with scheduler noise; object-heavy baseline timing is particularly affected by
tracing. Do not generalize these ratios to other workloads or infer that disabling
paths is slower from this small timing difference. The large run was measured
separately. No large legacy comparison was attempted.

The repeated-run test warms JIT, runs and discards five 4,000-history results,
forces collection, and checks tracked peak below 12 MiB and remaining tracked
allocation below 2 MiB. Returned paths are checked for owning storage. This finds
no new retention in the tested scenario; it is not a proof against every leak.

## Compatibility, guarantees and known limitations

- `max_samples_per_gen=40` is the default visualization mode; 0 is performance
  mode. Top-level keys, generation tuple layout, counts and histogram behavior
  remain compatible. Low-level transport now returns sampled paths rather than
  one path per input, uses owning contiguous arrays, and copies caller arrays.
- Sample indices are evenly spaced, deterministic, and independent of simulation
  randomness. The display sample is not a statistically random subset.
- Repeatability is guaranteed only for the same configuration, backend, platform
  and dependency versions as tested. Numba and NumPy have distinct RNG states.
  No bitwise cross-platform, cross-version, cross-backend, or legacy-sequence
  guarantee is claimed. Standalone samplers use their backend's RNG; use the seeded
  simulation interfaces for reproducible runs.
- Cancellation is checked between generations and before returning. A generation
  and JIT compilation cannot be interrupted. Closing requests shutdown without a
  blocking GUI join; the non-daemon worker keeps the process alive until it exits.
  Tests verify cooperative worker exit; arbitrary callbacks can still block.
- Numba transport releases the GIL. The fallback stays correct and is tested, but
  CPU contention and responsiveness there have not been benchmarked interactively.
- Memory remains proportional to particle/secondary populations, retained fission
  points, generations, and sample_count * (max_steps+1). This is bounded trajectory
  recording, not a promise that arbitrary configurations fit available RAM. Huge
  user-supplied populations, multiplicities or recording limits can still exhaust
  memory. There is no automatic memory-budget admission control.
- Callbacks execute on the simulation caller's thread. Applications must not
  supply GUI-mutating callbacks to a background simulation. Application transport
  calls are serialized; unrelated external users of NumPy global RNG are outside
  the fallback lock's scope.
- Existing educational interpretations and plots are preserved. No scientific
  model validation or new reactor-physics feature was attempted.

## Recommended next steps

1. Exercise the native Matplotlib backend on the target desktop through Run,
   animation, cancellation, close, failure and restart; measure event-loop latency.
2. If needed, add shorter cooperative transport checkpoints while preserving RNG
   order and numerical behavior, to reduce cancellation latency.
3. Consider a configurable memory budget and pre-binned heatmaps for very large
   generation counts. Preserve existing output compatibility when doing so.
4. Keep dependency versions recorded with reproducibility-sensitive results and
   expand benchmark repetitions before making general performance claims.

