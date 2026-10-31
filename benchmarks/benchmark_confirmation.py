"""Alternating paired timing confirmation without tracing or other benchmark jobs."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import types

parser = argparse.ArgumentParser()
parser.add_argument('--fallback', action='store_true')
parser.add_argument('--baseline-source', default='benchmarks/results/baseline_physics.txt',
                    help='Exact pre-change physics.py snapshot to compare')
parser.add_argument('--output', help='Results path; defaults to the historical confirmation path')
args = parser.parse_args()
if args.fallback:
    sys.modules['numba'] = None
from material import Material

modules = []
for name, path in [('confirm_baseline', args.baseline_source),
                   ('confirm_current', 'physics.py')]:
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(Path(path).read_bytes(), path, 'exec'), module.__dict__)
    modules.append(module)
records = []
for workload, xs in [('fission', (1., 0., 0.)), ('absorption', (0., 1., 0.)),
                      ('mixed', (.35, .25, .4))]:
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
        assert results[0]['k_history'] == results[1]['k_history']
        for a, b in zip(results[0]['generation_stats'], results[1]['generation_stats']):
            assert a['processed'] == b['processed']
            assert a['outcome_counts'] == b['outcome_counts']
        del results
    records.append(dict(workload=workload, n=20000, baseline_seconds=samples[0],
                        current_seconds=samples[1], baseline_median=statistics.median(samples[0]),
                        current_median=statistics.median(samples[1])))
backend = 'fallback' if args.fallback else 'numba'
Path(args.output or f'benchmarks/results/confirmation_{backend}.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
