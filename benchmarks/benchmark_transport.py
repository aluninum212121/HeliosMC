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
args = parser.parse_args()
if args.fallback:
    sys.modules['numba'] = None
import numpy as np
from material import Material

source_path = Path('benchmarks/results/baseline_physics.txt' if args.variant == 'baseline' else 'physics.py')
source = source_path.read_bytes()
module = types.ModuleType('measured_physics')
sys.modules[module.__name__] = module
exec(compile(source, str(source_path), 'exec'), module.__dict__)
try:
    import numba
    numba_version = numba.__version__
except ImportError:
    numba_version = None

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
