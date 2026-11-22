# HeliosMC: reliability and performance measurements

Historical measurements from 2026-10-30. These results describe the recorded
source snapshots and hardware, not fresh timings of every subsequent revision.

The pre-upgrade source is retained as
`benchmarks/results/baseline_physics.txt`, SHA256
`98d5df40ee21d46dde248d8902c3eeb5aa22db0f761ea48035764e478475caa4`.
The snapshot supports benchmarks and exact compatibility regressions;
it is not an application module.

## Estimator and completion contract


The existing arithmetic reconciles all outcomes, but arithmetic consistency does
not establish a physically valid estimator. A step-limited particle has no known
terminal interaction. Its unresolved future can affect both terminal counts and
the next source. The compatible source-production ratio remains

`k_g = floor(F_g * nu) / N_g`,

where `N_g` includes all transported histories and the numerator precedes
population capping. No unfinished histories were removed from the denominator.
Multiplicity truncation, site selection, direction law, transport formulas, event
order and population control are unchanged. The existing burn-in rule uses the
tail only if there are more than `n_burnin` generations; otherwise it uses all
available values. `k_std` is their population standard deviation, not a standard
error, convergence test, or physically validated uncertainty.

Generation dictionaries add `completed`, `step_limited`, `histories_complete`.
Completed is fission + absorption + leakage; completed + step_limited = processed.
The compatible `unfinished` outcome key and all tuple lengths are retained.
Results add execution success, aggregate completion counts, completeness and a
termination reason. `no_secondary_source` does not assert physical extinction if
there are unresolved histories. Empty runs retain the compatible zero summaries
but have no estimate. Failures still raise; cancellation still raises
SimulationCancelled. Neither returns a misleading completed partial result.

The GUI reports the compatible value as a source ratio, labels the generation
spread explicitly, warns if **any** generation has unfinished histories, and
withholds criticality classification and configuration advice in that case.
Empty runs display no estimate. Native desktop event-loop responsiveness was not measured in this benchmark.

## Focused implementation changes

1. Batch secondary-site index draws and gather positions for standard NumPy
   Generators. Remove zero-initialization of positions that are fully overwritten.
   Draw directions first as before. Custom RNG objects retain scalar calls.
2. Remove the duplicate `_arrays` scan in generation orchestration. The public
   transport boundary still validates arrays and copies before kernel mutation.
   No unvalidated public path was introduced.
3. Record a growing chronological list of fission **indices** inside transport,
   then restore owning coordinate arrays at the public boundary. A fission
   particle is inactive and its final position is unchanged after the event.
   This removes per-event coordinate allocations in the kernel without reserving
   an N-sized event buffer when there are no fissions. A separate alternating
   full-run prototype measured a 12% fission-workload time reduction and 17%
   tracked peak reduction relative to the source-batching stage; mixed also
   improved and scatter-only peak was unchanged. Public result types, order and
   ownership are regression-tested against the original source.
4. Add optional appended `max_memory_bytes` to the three public simulation
   functions. Default: 512 MiB; `None` disables it. GUI exposes the budget and
   max_steps as programmatic settings. Before source allocation, transport scratch
   and secondary expansion, estimate `512*N + 48*min(N,S)*(steps+1) + 128*N_secondary`
   bytes, deducting retained visualization array bytes from subsequent generation
   budgets. This deliberately conservative planning guard raises MemoryError.
   It is not a hard process-RSS cap and does not cover interpreter/JIT/allocator
   overhead, external caller storage, every Python container or all native memory.
   A nonfinite secondary count also fails before attempting allocation.

The full secondary population is still materialized before uniform population
control. Generating only the capped population would change RNG consumption and
potentially the source-selection contract; it was not implemented. No scientific
assumption or seeded-stream change was required or approved.

## Benchmark methodology and environment

- Windows 11, build 26200; Intel Core i5-8250U, 4 cores / 8 logical processors;
  physical RAM 8,475,860,992 bytes (about 7.89 GiB).
- Python 3.14.4 (64-bit MSC v.1944); NumPy 2.4.4; Numba 0.67.0;
  Matplotlib 3.10.9. No dependency was installed for the upgrade.
- Seed 2026; N = 150, 2,000, 10,000, 20,000. The largest population was chosen
  conservatively for this 8 GB machine. Eight transport steps, zero sampled paths.
- Synthetic scatter-only, mixed `(f,a,s)=(.35,.25,.4)`, fission-only,
  absorption-only and leakage-heavy workloads. nu=2.43 for branching workloads;
  radius=1e9 except leakage radius=1e-12. Density and molar mass are both 1.
- Each generation benchmark uses fixed arrays; end-to-end uses up to three
  dependent generations with the existing cap. Scatter-only histories are
  intentionally incomplete; absorption/leakage extinguish the source.
- Three untraced `perf_counter` repetitions, reporting median and raw times.
  Match scalar signatures with a small warm generation before measurement.
  Record the first transport call separately, including JIT compilation/startup.
  Timing and memory runs are distinct; a fourth traced run records retained and
  peak tracked allocations. Results are released between timed repetitions.
- Secondary isolation stubs transport deterministically with unique fission
  positions. This includes public orchestration, result conversion and stub event
  list construction; it is explicitly **not** pure RNG/gather kernel timing.
- Numba and forced no-Numba runs use separate fresh processes. Raw JSON records
  preserve configuration, backend, source hash, outcome counts, processed counts,
  completion status, k histories and all individual measurements.
- The layout experiment uses contiguous component columns (Fortran `(N,3)`,
  equivalent to a `(3,N)` contiguous backing allocation), preserving the same
  indexed kernel. It includes input layout conversion and C-contiguous output
  adaptation. Five timed repetitions plus separate tracing. This is one practical
  SoA candidate, not a claim about every possible six-array kernel.
- A separate candidate preallocates chronological fission storage, then restores
  the public list-of-owning-arrays contract. This measures the compatibility cost
  rather than silently changing the result type.

The historical harness commands below write the named benchmark outputs.
Use a separate working copy to retain the published measurements unchanged.
Run from its repository root with a Numba-capable interpreter:

```powershell
py -3.14 -m benchmarks.benchmark_transport --variant baseline --output benchmarks/results/baseline_numba.json
py -3.14 -m benchmarks.benchmark_transport --variant baseline --fallback --output benchmarks/results/baseline_fallback.json
py -3.14 -m benchmarks.benchmark_candidates
py -3.14 -m benchmarks.benchmark_event_indices
py -3.14 -m benchmarks.benchmark_transport --variant current --output benchmarks/results/current_numba.json
py -3.14 -m benchmarks.benchmark_transport --variant current --fallback --output benchmarks/results/current_fallback.json
py -3.14 -m benchmarks.benchmark_confirmation
py -3.14 -m benchmarks.benchmark_confirmation --fallback
py -3.14 -m benchmarks.benchmark_report
py -3.14 -m pytest -q -p no:cacheprovider
```

Measurement limits: tracemalloc reports tracked allocations rather than total
process memory. JIT and allocator memory may not be represented. The first call
is compilation-inclusive elapsed time, not an isolated compiler profiler. These
are local laptop observations; power mode, thermal state and background load were
not controlled. Baseline benchmark work overlapped some validation work; benchmark
jobs themselves were sequential. Treat small timing differences as inconclusive.
Only robust end-to-end benefits warrant retention. Earlier v0.2 measurements were not substituted for this dataset.

`benchmarks/results/secondary_stage_physics.txt` and `secondary_stage_*.json` preserve the intermediate
source-batching stage (physics hash
`c2d5d5c02be59297efd5bb6337e9678de52117e1c1b326900a94a64a16b81939`).
The integer-event experiment uses that fixed source, so it remains reproducible
after adoption. Final `current_*.json` tables measure the final implementation.
The initial matrix's load-related timing variation motivated five-repeat
alternating timing-only confirmation, with benchmark/test jobs run sequentially.

## Testing and reproducibility

Exact compatibility tests use the frozen baseline source, checking all five
standard NumPy bit generators, empty/singleton/odd/large source sizes, source
positions/directions and subsequent RNG bytes. A full seeded multi-generation run
compares ratios, summary values, counts, chronological events, sampled paths and
population caps. Tests also cover custom scalar RNG objects, noncontiguous arrays,
float32/integer inputs, input immutability, memory rejection before source
allocation, secondary growth, retained frames, completion states and GUI warnings.
The worker's existing failure/cancellation tests remain in the full suite.

Final forced-fallback targeted checks passed 40 tests (7 deselected) in 1.19 seconds,
including baseline equivalence and RNG continuation. The existing suite also
launches a separate no-Numba process. Cross-backend, cross-version and cross-platform
bitwise equivalence is not promised. The fallback still saves/restores global
legacy NumPy RNG state under the transport lock; unrelated external global RNG
users need not honor that lock. Cancellation timing remains nondeterministic.

The historical final integration suite passed **83 tests and 2 subtests**
in 27.04 seconds. These counts describe that revision. Use the root README
for current test commands; later tests are not retroactive benchmark evidence.

## Remaining issues and next work

Retention decisions: keep source batching and chronological integer events, whose
benefits survive end-to-end checks. Keep the current `(N,3)` layout: component
layout timings were inconsistent across workloads and increased scatter-only
tracked peak (1.470 to 1.909 MiB at N=20,000). Reject the preallocated event buffer:
although useful for high fission rates, it increased scatter-only peak to 1.928
MiB. Integer event lists achieved useful branching-memory reductions without
that reservation. No reusable random buffers or active-index transport rewrite
was adopted without evidence of compatible end-to-end benefit.

- No parallel transport or active-particle compaction: both require deliberate
  RNG/data-ownership design. Dead particles still consume distance/reaction draws.
- Reusable random work buffers were inspected but not introduced: legacy backend
  RNG helpers return newly allocated arrays and have no output-buffer argument.
  Fusing loops or drawing only for active particles would alter the draw schedule.
  The transport equations and loop order were kept intact.
- Fission points and sampled frames still accumulate across generations. The
  storage planning guard bounds configured growth conservatively, not process RSS.
  A future streaming/retention API should explicitly define visualization contracts.
- Public array coercion/validation can allocate before the transport storage
  check; the budget does not bound caller-owned arrays or every boundary temporary.
  The main `estimate_k_full` source allocation is checked before allocation.
- Full secondary allocation before capping remains a limitation. A separately
  reviewed sampling/RNG contract is needed before changing it.
- Cancellation waits for the current generation, including first-use compilation.
  Worker BaseException behavior identified in the audit is unchanged.
- Native GUI latency, broader sample-count/max-step matrices, allocator profiling,
  controlled paired laptop runs and other hardware remain useful follow-up work.
- Physical estimator/uncertainty validation requires domain review.
  Current material metadata and sources are described in ARCHITECTURE_AUDIT.md;
  software regression tests do not establish physical validity.

Measured comparison and final integration results are recorded below.

## Measured comparison

All differences below are `(optimized - baseline) / baseline`. Negative time differences mean faster execution. Memory is peak tracked allocation, not total RSS. Values are medians of three untraced repetitions.

### Numba

Compilation-inclusive first transport call: 19.168905 s baseline; 5.661317 s current. These first-call times are excluded from the steady-state tables.

| N | Stage / workload | Baseline ms | Optimized ms | Time difference | Baseline peak MiB | Optimized peak MiB | Memory difference |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150 | generation / scatter | 1.550 | 0.332 | -78.6% | 0.015 | 0.015 | -0.4% |
| 150 | end_to_end / scatter | 2.153 | 0.697 | -67.6% | 0.024 | 0.024 | +0.3% |
| 150 | generation / mixed | 4.052 | 0.686 | -83.1% | 0.057 | 0.042 | -25.4% |
| 150 | end_to_end / mixed | 13.748 | 1.533 | -88.9% | 0.083 | 0.068 | -17.8% |
| 150 | generation / fission | 7.025 | 0.438 | -93.8% | 0.088 | 0.064 | -27.6% |
| 150 | end_to_end / fission | 24.851 | 1.644 | -93.4% | 0.124 | 0.100 | -19.8% |
| 150 | generation / absorption | 1.621 | 0.222 | -86.3% | 0.015 | 0.015 | -0.4% |
| 150 | end_to_end / absorption | 1.647 | 0.448 | -72.8% | 0.024 | 0.024 | -0.1% |
| 150 | generation / leak | 1.139 | 0.274 | -75.9% | 0.015 | 0.015 | -0.4% |
| 150 | end_to_end / leak | 2.185 | 0.446 | -79.6% | 0.024 | 0.024 | -0.1% |
| 150 | secondary_with_wrapper / fission | 4.701 | 0.340 | -92.8% | 0.057 | 0.052 | -9.2% |
| 2,000 | generation / scatter | 6.105 | 2.773 | -54.6% | 0.151 | 0.151 | -0.0% |
| 2,000 | end_to_end / scatter | 10.486 | 3.004 | -71.4% | 0.259 | 0.259 | -0.0% |
| 2,000 | generation / mixed | 76.354 | 2.798 | -96.3% | 0.695 | 0.512 | -26.3% |
| 2,000 | end_to_end / mixed | 168.835 | 8.088 | -95.2% | 1.001 | 0.818 | -18.3% |
| 2,000 | generation / fission | 71.270 | 2.637 | -96.3% | 1.071 | 0.800 | -25.3% |
| 2,000 | end_to_end / fission | 362.205 | 11.448 | -96.8% | 1.510 | 1.239 | -18.0% |
| 2,000 | generation / absorption | 1.781 | 0.335 | -81.2% | 0.151 | 0.151 | +0.0% |
| 2,000 | end_to_end / absorption | 2.066 | 0.817 | -60.4% | 0.259 | 0.259 | +0.0% |
| 2,000 | generation / leak | 2.733 | 0.473 | -82.7% | 0.151 | 0.151 | +0.0% |
| 2,000 | end_to_end / leak | 2.515 | 0.878 | -65.1% | 0.259 | 0.259 | +0.0% |
| 2,000 | secondary_with_wrapper / fission | 70.933 | 2.091 | -97.1% | 0.666 | 0.651 | -2.3% |
| 10,000 | generation / scatter | 29.553 | 9.487 | -67.9% | 0.738 | 0.738 | +0.0% |
| 10,000 | end_to_end / scatter | 31.994 | 11.646 | -63.6% | 1.273 | 1.273 | +0.0% |
| 10,000 | generation / mixed | 248.522 | 11.869 | -95.2% | 3.211 | 2.500 | -22.1% |
| 10,000 | end_to_end / mixed | 589.275 | 39.806 | -93.2% | 4.765 | 4.048 | -15.0% |
| 10,000 | generation / fission | 259.096 | 14.051 | -94.6% | 5.209 | 3.975 | -23.7% |
| 10,000 | end_to_end / fission | 1040.380 | 53.516 | -94.9% | 7.393 | 6.159 | -16.7% |
| 10,000 | generation / absorption | 4.470 | 1.117 | -75.0% | 0.738 | 0.738 | +0.0% |
| 10,000 | end_to_end / absorption | 4.815 | 1.828 | -62.0% | 1.273 | 1.273 | +0.0% |
| 10,000 | generation / leak | 2.378 | 1.049 | -55.9% | 0.738 | 0.738 | +0.0% |
| 10,000 | end_to_end / leak | 6.694 | 2.240 | -66.5% | 1.273 | 1.273 | +0.0% |
| 10,000 | secondary_with_wrapper / fission | 355.614 | 10.670 | -97.0% | 3.187 | 3.241 | +1.7% |
| 20,000 | generation / scatter | 47.608 | 23.374 | -50.9% | 1.473 | 1.473 | +0.0% |
| 20,000 | end_to_end / scatter | 51.880 | 20.417 | -60.6% | 2.542 | 2.542 | +0.0% |
| 20,000 | generation / mixed | 355.638 | 24.930 | -93.0% | 6.511 | 5.061 | -22.3% |
| 20,000 | end_to_end / mixed | 1034.521 | 122.684 | -88.1% | 9.509 | 8.071 | -15.1% |
| 20,000 | generation / fission | 568.278 | 55.925 | -90.2% | 10.415 | 7.945 | -23.7% |
| 20,000 | end_to_end / fission | 1956.303 | 91.121 | -95.3% | 14.779 | 12.309 | -16.7% |
| 20,000 | generation / absorption | 4.472 | 1.583 | -64.6% | 1.473 | 1.473 | +0.0% |
| 20,000 | end_to_end / absorption | 9.014 | 3.600 | -60.1% | 2.542 | 2.542 | +0.0% |
| 20,000 | generation / leak | 5.286 | 1.860 | -64.8% | 1.473 | 1.473 | +0.0% |
| 20,000 | end_to_end / leak | 6.011 | 3.531 | -41.3% | 2.542 | 2.542 | +0.0% |
| 20,000 | secondary_with_wrapper / fission | 354.571 | 22.412 | -93.7% | 6.371 | 6.479 | +1.7% |

Every paired processed count, outcome count, completion status, secondary count and generation ratio in this matrix agrees exactly.

### Fallback

Compilation-inclusive first transport call: 0.040705 s baseline; 0.019128 s current. There is no JIT in this backend.

| N | Stage / workload | Baseline ms | Optimized ms | Time difference | Baseline peak MiB | Optimized peak MiB | Memory difference |
|---:|---|---:|---:|---:|---:|---:|---:|
| 150 | generation / scatter | 46.837 | 16.609 | -64.5% | 0.020 | 0.020 | -0.4% |
| 150 | end_to_end / scatter | 41.307 | 15.116 | -63.4% | 0.028 | 0.028 | +0.1% |
| 150 | generation / mixed | 5.499 | 2.611 | -52.5% | 0.046 | 0.042 | -7.2% |
| 150 | end_to_end / mixed | 37.661 | 9.604 | -74.5% | 0.071 | 0.068 | -4.6% |
| 150 | generation / fission | 5.027 | 1.197 | -76.2% | 0.069 | 0.064 | -7.4% |
| 150 | end_to_end / fission | 20.980 | 4.499 | -78.6% | 0.104 | 0.099 | -4.9% |
| 150 | generation / absorption | 3.770 | 1.419 | -62.4% | 0.019 | 0.019 | +0.0% |
| 150 | end_to_end / absorption | 4.537 | 1.188 | -73.8% | 0.028 | 0.028 | -0.0% |
| 150 | generation / leak | 7.564 | 0.982 | -87.0% | 0.019 | 0.019 | +0.0% |
| 150 | end_to_end / leak | 4.385 | 1.249 | -71.5% | 0.028 | 0.028 | +0.1% |
| 150 | secondary_with_wrapper / fission | 2.807 | 0.245 | -91.3% | 0.057 | 0.052 | -9.2% |
| 2,000 | generation / scatter | 939.397 | 198.977 | -78.8% | 0.177 | 0.177 | -0.0% |
| 2,000 | end_to_end / scatter | 845.336 | 180.489 | -78.6% | 0.284 | 0.284 | +0.0% |
| 2,000 | generation / mixed | 237.579 | 31.390 | -86.8% | 0.544 | 0.512 | -5.9% |
| 2,000 | end_to_end / mixed | 661.986 | 100.272 | -84.9% | 0.850 | 0.818 | -3.8% |
| 2,000 | generation / fission | 54.114 | 10.753 | -80.1% | 0.812 | 0.800 | -1.5% |
| 2,000 | end_to_end / fission | 149.692 | 34.325 | -77.1% | 1.251 | 1.239 | -1.0% |
| 2,000 | generation / absorption | 36.432 | 8.861 | -75.7% | 0.162 | 0.162 | -0.0% |
| 2,000 | end_to_end / absorption | 28.573 | 9.523 | -66.7% | 0.269 | 0.269 | +0.0% |
| 2,000 | generation / leak | 29.272 | 8.377 | -71.4% | 0.162 | 0.162 | -0.0% |
| 2,000 | end_to_end / leak | 29.357 | 8.893 | -69.7% | 0.269 | 0.269 | +0.0% |
| 2,000 | secondary_with_wrapper / fission | 60.862 | 1.443 | -97.6% | 0.666 | 0.651 | -2.3% |
| 10,000 | generation / scatter | 4185.994 | 881.752 | -78.9% | 0.856 | 0.856 | -0.0% |
| 10,000 | end_to_end / scatter | 3456.933 | 964.437 | -72.1% | 1.391 | 1.391 | +0.0% |
| 10,000 | generation / mixed | 833.877 | 154.864 | -81.4% | 2.466 | 2.500 | +1.4% |
| 10,000 | end_to_end / mixed | 1945.007 | 493.733 | -74.6% | 4.013 | 4.048 | +0.9% |
| 10,000 | generation / fission | 415.012 | 53.369 | -87.1% | 3.917 | 3.975 | +1.5% |
| 10,000 | end_to_end / fission | 1258.005 | 165.871 | -86.8% | 6.101 | 6.158 | +0.9% |
| 10,000 | generation / absorption | 149.799 | 44.438 | -70.3% | 0.779 | 0.779 | -0.0% |
| 10,000 | end_to_end / absorption | 129.217 | 45.931 | -64.5% | 1.314 | 1.314 | +0.0% |
| 10,000 | generation / leak | 108.725 | 42.454 | -61.0% | 0.779 | 0.779 | -0.0% |
| 10,000 | end_to_end / leak | 92.474 | 42.506 | -54.0% | 1.314 | 1.314 | +0.0% |
| 10,000 | secondary_with_wrapper / fission | 269.388 | 7.368 | -97.3% | 3.187 | 3.241 | +1.7% |
| 20,000 | generation / scatter | 4300.071 | 1747.197 | -59.4% | 1.704 | 1.704 | +0.0% |
| 20,000 | end_to_end / scatter | 3190.315 | 1855.318 | -41.8% | 2.773 | 2.773 | +0.0% |
| 20,000 | generation / mixed | 596.456 | 341.215 | -42.8% | 4.994 | 5.061 | +1.3% |
| 20,000 | end_to_end / mixed | 2194.144 | 1047.078 | -52.3% | 8.004 | 8.071 | +0.8% |
| 20,000 | generation / fission | 971.314 | 108.271 | -88.9% | 7.833 | 7.945 | +1.4% |
| 20,000 | end_to_end / fission | 2623.782 | 352.451 | -86.6% | 12.197 | 12.309 | +0.9% |
| 20,000 | generation / absorption | 230.334 | 91.612 | -60.2% | 1.551 | 1.551 | -0.0% |
| 20,000 | end_to_end / absorption | 245.692 | 91.147 | -62.9% | 2.620 | 2.620 | +0.0% |
| 20,000 | generation / leak | 223.915 | 85.117 | -62.0% | 1.551 | 1.551 | -0.0% |
| 20,000 | end_to_end / leak | 261.098 | 89.457 | -65.7% | 2.620 | 2.620 | +0.0% |
| 20,000 | secondary_with_wrapper / fission | 403.700 | 25.314 | -93.7% | 6.371 | 6.479 | +1.7% |

Every paired processed count, outcome count, completion status, secondary count and generation ratio in this matrix agrees exactly.

### Layout and fission-storage candidates

Five untraced repetitions per candidate; adapters included. No production layout change or preallocated event buffer has been retained. A subsequent chronological integer-list candidate was measured separately.

| N | Workload | Existing ms | Component layout ms | Preallocated events ms | Existing / component / event peak MiB |
|---:|---|---:|---:|---:|---|
| 150 | scatter | 0.513 | 0.497 | 0.487 | 0.012 / 0.015 / 0.016 |
| 150 | mixed | 0.440 | 0.445 | 0.408 | 0.037 / 0.039 / 0.025 |
| 150 | fission | 0.436 | 0.348 | 0.253 | 0.058 / 0.058 / 0.033 |
| 2,000 | scatter | 2.905 | 2.545 | 2.866 | 0.148 / 0.192 / 0.194 |
| 2,000 | mixed | 2.475 | 2.861 | 2.246 | 0.478 / 0.504 / 0.308 |
| 2,000 | fission | 1.866 | 1.623 | 2.781 | 0.764 / 0.764 / 0.421 |
| 10,000 | scatter | 23.727 | 19.156 | 14.856 | 0.736 / 0.955 / 0.965 |
| 10,000 | mixed | 12.913 | 10.590 | 11.270 | 2.353 / 2.492 / 1.518 |
| 10,000 | fission | 12.348 | 12.095 | 6.247 | 3.792 / 3.792 / 2.104 |
| 20,000 | scatter | 22.606 | 24.203 | 22.783 | 1.470 / 1.909 / 1.928 |
| 20,000 | mixed | 37.004 | 22.572 | 17.469 | 4.901 / 5.035 / 3.061 |
| 20,000 | fission | 24.763 | 22.855 | 15.486 | 7.555 / 7.555 / 4.210 |

Candidate verification: 20 generator/bound pairs have exact scalar/batch index and continuation agreement. All layout/event candidate output positions, directions, fates and chronological fission events matched the reference for the measured cases.

### Alternating timing confirmation

To reduce the load/order confounding visible in the first matrix, a separate five-repeat timing-only confirmation alternated baseline/current order at N=20,000, three generations. No tracing or other benchmark/test jobs ran concurrently. These are the preferred timing comparisons; individual repetitions remain in confirmation JSON files.

| Backend | Workload | Baseline ms | Optimized ms | Speed ratio |
|---|---|---:|---:|---:|
| numba | fission | 577.046 | 92.498 | 6.24× |
| numba | absorption | 3.993 | 2.918 | 1.37× |
| numba | mixed | 344.596 | 98.340 | 3.50× |
| fallback | fission | 830.127 | 361.729 | 2.29× |
| fallback | absorption | 112.201 | 105.865 | 1.06× |
| fallback | mixed | 1411.520 | 1129.638 | 1.25× |

The branching benefit is confirmed. Small nonbranching differences remain subject to substantial run-to-run variation. Earlier absolute timings were affected by load, so do not interpret their larger speed ratios as portable performance promises.

### Integer event-storage refinement

The retained refinement was tested against the frozen source-batching stage, with five alternating timing repetitions and separate tracing. Counts, ratios and chronological event coordinates matched exactly.

| Workload, N=20,000 | Before ms | Integer events ms | Before peak MiB | Integer events peak MiB |
|---|---:|---:|---:|---:|
| scatter | 36.440 | 26.633 | 2.539 | 2.539 |
| mixed | 133.258 | 112.316 | 9.569 | 8.067 |
| fission | 155.922 | 137.433 | 14.884 | 12.305 |
