"""Compare chronological integer event storage with current owning event copies."""
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import types

from material import Material
import numpy as np

source = Path('benchmarks/results/secondary_stage_physics.txt').read_text(encoding='utf-8-sig')
candidate_source = source.replace('fission_positions.append(positions[i].copy())',
                                  'fission_positions.append(i)')
candidate_source = candidate_source.replace('pos, dirs, fates, fiss, paths, lengths = out',
    'pos, dirs, fates, fiss, paths, lengths = out\n    fiss = [pos[i].copy() for i in fiss]')
modules = []
for name, code in [('event_before', source), ('event_indices', candidate_source)]:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(code, name, 'exec'), module.__dict__)
    modules.append(module)

records = []
for workload, xs in [('scatter', (0., 0., 1.)), ('mixed', (.35, .25, .4)),
                      ('fission', (1., 0., 0.))]:
    mat = Material(workload, *xs, 2.43, 1., 1., 'red')
    config = dict(n_gen=3, n_source=20000, radius=1e9, material=mat,
                  seed=2026, max_steps=8, max_samples_per_gen=0)
    for module in modules:
        module.estimate_k_full(**(config | {'n_source': 2}))
    samples = [[], []]
    for repeat in range(5):
        results = [None, None]
        for i in ([0, 1] if repeat % 2 == 0 else [1, 0]):
            start = time.perf_counter()
            results[i] = modules[i].estimate_k_full(**config)
            samples[i].append(time.perf_counter() - start)
        assert results[0]['generation_stats'] == results[1]['generation_stats']
        assert results[0]['k_history'] == results[1]['k_history']
        for a, b in zip(results[0]['fiss_frames'], results[1]['fiss_frames']):
            np.testing.assert_array_equal(a, b)
        del results
    peaks = []
    for module in modules:
        tracemalloc.start()
        result = module.estimate_k_full(**config)
        _, peak = tracemalloc.get_traced_memory()
        peaks.append(peak)
        tracemalloc.stop()
        del result
    records.append(dict(workload=workload, n=20000, seconds=samples,
                        median_seconds=[statistics.median(s) for s in samples],
                        peak_tracked_bytes=peaks))
Path('benchmarks/results/event_indices.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
