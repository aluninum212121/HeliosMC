"""Reproducible paired benchmark against the pre-upgrade working-tree snapshot.

Timing excludes tracing; a separate invocation measures tracked allocations.
No historical Git code is executed. Run from the repository root.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
import types

parser = argparse.ArgumentParser()
parser.add_argument('--variant', choices=['baseline', 'current'], required=True)
parser.add_argument('--fallback', action='store_true')
parser.add_argument('--output', required=True)
parser.add_argument('--educational', action='store_true', help='Compare abstract absorption and preserved legacy path')
parser.add_argument('--baseline-source', default='benchmarks/fixtures/baseline_physics.txt')
args = parser.parse_args()
if args.fallback:
    sys.modules['numba'] = None
import numpy as np
from material import Material

source_path = Path('benchmarks/fixtures/baseline_physics.txt' if args.variant == 'baseline' else 'physics.py')
source = source_path.read_bytes()
module = types.ModuleType('measured_physics')
sys.modules[module.__name__] = module
exec(compile(source, str(source_path), 'exec'), module.__dict__)
try:
    import numba
    numba_version = numba.__version__
except ImportError:
    numba_version = None


def benchmark_educational():
    """Same analytic attenuation problem; distinct RNG streams across models."""
    import scipy
    from material import MaterialDataset, MaterialQuantity

    baseline_source = Path(args.baseline_source).read_bytes()
    baseline = types.ModuleType('educational_benchmark_baseline')
    sys.modules[baseline.__name__] = baseline
    exec(compile(baseline_source, args.baseline_source, 'exec'), baseline.__dict__)
    medium = MaterialDataset('benchmark:abstract', 'Synthetic attenuation', 'synthetic',
        'analytic benchmark', '1', ('Invented unit attenuation per cm.',),
        (MaterialQuantity('absorption_rate', '1/cm', 1.), MaterialQuantity('scattering_rate', '1/cm', 0.)))
    module.summarize_binomial(1, 2)  # Separate SciPy startup from first JIT call.
    cold = {}
    for name, target in [('legacy_before', baseline), ('legacy_current', module)]:
        start = time.perf_counter()
        target.transport_batch(np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)),
                               1., 0., 1., 0., 0., 2., max_steps=1, max_samples=0, seed=2026)
        cold[name] = time.perf_counter() - start
    start = time.perf_counter()
    module._slab_batch(np.array([0., 2.]), np.array([1.]), np.array([0.]),
                      2, 1, np.random.default_rng(2026), 0, 0., 1.)
    cold['slab_kernel'] = time.perf_counter() - start
    records = []
    for n in (150, 2000, 20000):
        for sample_count in (0, 40):
            p, d = np.zeros((n, 3)), np.tile([0., 0., 1.], (n, 1))
            def legacy(target):
                r = target.transport_batch(p, d, 1., 0., 1., 0., 0., 2., max_steps=1,
                                           max_samples=sample_count, seed=2026)
                return dict(absorbed=int(np.sum(r[2] == 1)), escaped=int(np.sum(r[2] == 2)))
            def slab(backend):
                return module.simulate_slab([medium], [0., 2.], n_histories=n,
                    max_collisions=1, max_samples=sample_count, seed=2026, backend=backend)
            functions = {'legacy_before': lambda: legacy(baseline),
                         'legacy_current': lambda: legacy(module),
                         'slab_python_numpy': lambda: slab('python')}
            if module.HAS_NUMBA:
                functions['slab_numba'] = lambda: slab('numba')
            for function in functions.values():
                function()  # Warm exact sample/signature and statistical paths.
            durations = {name: [] for name in functions}
            results = {}
            for repeat in range(5):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    start = time.perf_counter()
                    results[name] = functions[name]()
                    durations[name].append(time.perf_counter() - start)
            assert results['legacy_before'] == results['legacy_current']
            if module.HAS_NUMBA:
                assert results['slab_numba']['outcome_counts'] == results['slab_python_numpy']['outcome_counts']
            for name, function in functions.items():
                gc.collect()
                tracemalloc.start()
                result = function()
                retained, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                records.append(dict(variant=name, n=n, samples=sample_count,
                    seconds=durations[name], median_seconds=statistics.median(durations[name]),
                    peak_tracked_bytes=peak, retained_tracked_bytes=retained,
                    outcomes=result.get('outcome_counts', result),
                    timing=result.get('timing')))
    process_peak = None
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                ('peak_ws', 'ws', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        process_peak = counters.peak_ws
    output = dict(workload='straight_absorption_rate_1_width_2', seed=2026, repeats=5,
        expected_transmission=float(np.exp(-2.)), cold_first_call_seconds=cold,
        cold_note='includes first dispatch and JIT compilation when enabled; excludes SciPy startup',
        process_peak_working_set_bytes=process_peak,
        memory_note='tracemalloc per call; process working-set peak includes all variants/imports/JIT, not attributable per variant',
        source_sha256=hashlib.sha256(source).hexdigest(),
        baseline_sha256=hashlib.sha256(baseline_source).hexdigest(),
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__, numba=numba_version,
        platform=platform.platform(), backend='numba' if module.HAS_NUMBA else 'python', records=records)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'records'}, indent=2))


if args.educational:
    benchmark_educational()
    raise SystemExit(0)

SEED = 2026
REPEATS = 3
STEPS = 8
materials = {
    'scatter': Material('scatter', 0, 0, 1, 0, 1, 1, 'red'),
    'mixed': Material('mixed', .35, .25, .4, 2.43, 1, 1, 'red'),
    'fission': Material('fission', 1, 0, 0, 2.43, 1, 1, 'red'),
    'absorption': Material('absorption', 0, 1, 0, 0, 1, 1, 'red'),
    'leak': Material('leak', 0, 0, 1, 0, 1, 1, 'red'),
}

def measure(function):
    durations = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function()
        durations.append(time.perf_counter() - start)
        del result
    gc.collect()
    tracemalloc.start()
    result = function()
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(seconds=durations, median_seconds=statistics.median(durations),
                peak_tracked_bytes=peak, retained_bytes=retained), result

start = time.perf_counter()
module.transport_batch(np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)),
                       1., 1., 0., 0., 2.43, 1e9, max_steps=STEPS, max_samples=0, seed=SEED)
first_call = time.perf_counter() - start
records = []
for n in (150, 2000, 10000, 20000):
    for name, mat in materials.items():
        p = np.zeros((n, 3))
        d = np.tile([0., 0., 1.], (n, 1))
        radius = 1e-12 if name == 'leak' else 1e9
        config = dict(max_steps=STEPS, max_samples_per_gen=0, seed=SEED)
        # Warm each material's exact scalar signature before measuring.
        module.run_generation_batch(p[:2], d[:2], mat, radius, **config)
        timing, result = measure(lambda: module.run_generation_batch(p, d, mat, radius, **config))
        counts = result[-1]
        records.append(dict(stage='generation', n=n, workload=name, processed=sum(counts.values()),
                            status='incomplete' if counts['unfinished'] else 'complete',
                            counts=counts, **timing))
        del result
        timing, result = measure(lambda: module.estimate_k_full(n_gen=3, n_source=n, material=mat,
                                                                 radius=radius, **config))
        records.append(dict(stage='end_to_end', n=n, workload=name,
                            processed=sum(s['processed'] for s in result['generation_stats']),
                            k_history=result['k_history'],
                            status='incomplete' if any(s['outcome_counts']['unfinished'] for s in result['generation_stats']) else 'complete',
                            **timing))
        del result
    # Isolate secondary construction with deterministic transport output.
    fates = np.zeros(n, dtype=np.int32)
    final = np.arange(n * 3, dtype=float).reshape(n, 3)
    saved_transport = module.transport_batch
    module.transport_batch = lambda *a, **kw: (final, d, fates, list(final), [])
    timing, result = measure(lambda: module.run_generation_batch(p, d, materials['fission'], 1e9,
                                                                max_samples_per_gen=0, seed=SEED))
    module.transport_batch = saved_transport
    records.append(dict(stage='secondary_with_wrapper', n=n, workload='fission',
                        secondary=len(result[0]), **timing))

output = dict(variant=args.variant, backend='numba' if module.HAS_NUMBA else 'python',
              source_sha256=hashlib.sha256(source).hexdigest(), python=sys.version,
              numpy=np.__version__, numba=numba_version, platform=platform.platform(),
              processor=platform.processor(), logical_cpus=os.cpu_count(), seed=SEED,
              repeats=REPEATS, steps=STEPS, samples=0, first_transport_call_seconds=first_call,
              records=records)
Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in output.items() if k != 'records'}, indent=2))
