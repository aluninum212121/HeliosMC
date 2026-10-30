"""Rejected/retained layout and source candidates; includes conversion costs."""
import json
from pathlib import Path
import statistics
import sys
import time
import tracemalloc
import types

import numpy as np

source = Path('benchmarks/results/baseline_physics.txt').read_text(encoding='utf-8-sig')
reference = types.ModuleType('candidate_reference')
sys.modules[reference.__name__] = reference
exec(compile(source, '<candidate reference>', 'exec'), reference.__dict__)
# F-order (N,3) uses a contiguous (3,N) backing layout: x, y, z columns
# and ux, uy, uz columns each contiguous. Public results return to C order.
soa_source = source.replace('positions.copy(), directions.copy()',
                            'positions.copy(order="F"), directions.copy(order="F")')
soa_source = soa_source.replace('pos, dirs, fates, fiss, paths, lengths = out',
    'pos, dirs, fates, fiss, paths, lengths = out\n    pos = np.ascontiguousarray(pos)\n    dirs = np.ascontiguousarray(dirs)')
soa = types.ModuleType('candidate_soa')
sys.modules[soa.__name__] = soa
exec(compile(soa_source, '<candidate soa>', 'exec'), soa.__dict__)
event_source = source.replace('fission_positions = []',
    'fission_positions = np.empty((n_neutrons, 3), dtype=np.float64)\n    fission_count = 0')
event_source = event_source.replace('fission_positions.append(positions[i].copy())',
    'fission_positions[fission_count] = positions[i]\n                fission_count += 1')
event_source = event_source.replace('fates, fission_positions, paths, lengths',
                                   'fates, fission_positions[:fission_count], paths, lengths')
event_source = event_source.replace('pos, dirs, fates, fiss, paths, lengths = out',
    'pos, dirs, fates, fiss, paths, lengths = out\n    fiss = [row.copy() for row in fiss]')
events = types.ModuleType('candidate_events')
sys.modules[events.__name__] = events
exec(compile(event_source, '<candidate events>', 'exec'), events.__dict__)

def timed(fn):
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
        del result
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dict(seconds=samples, median_seconds=statistics.median(samples),
                peak_tracked_bytes=peak)

records = []
for n in (150, 2000, 10000, 20000):
    p = np.zeros((n, 3))
    d = np.tile([0., 0., 1.], (n, 1))
    for name, xs in [('scatter', (0., 0., 1.)), ('mixed', (.35, .25, .4)), ('fission', (1., 0., 0.))]:
        options = dict(max_steps=8, max_samples=0, seed=2026)
        def run(module):
            return module.transport_batch(p, d, 1., *xs, 2.43, 1e9, **options)
        a = run(reference)
        for variant in (soa, events):
            b = run(variant)  # compile/warm each signature
            for x, y in zip(a[:4], b[:4]):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        records.append(dict(n=n, workload=name, aos=timed(lambda: run(reference)),
                            soa=timed(lambda: run(soa)), events=timed(lambda: run(events))))

# Check supported standard NumPy generators; do not infer compatibility.
rng_cases = []
for bitgen in (np.random.PCG64, np.random.PCG64DXSM, np.random.MT19937,
               np.random.Philox, np.random.SFC64):
    for high in (1, 2, 17, 20000):
        a = np.random.Generator(bitgen(2026))
        b = np.random.Generator(bitgen(2026))
        x = np.array([a.integers(0, high) for _ in range(10001)])
        y = b.integers(0, high, size=10001)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(a.bytes(100), b.bytes(100))
        rng_cases.append(dict(generator=bitgen.__name__, high=high, exact=True))
Path('benchmarks/results/candidates.json').write_text(json.dumps(dict(layout=records, rng=rng_cases), indent=2))
print('Layout comparisons:', len(records), '; exact RNG comparisons:', len(rng_cases))
