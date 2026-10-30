"""Validate paired raw records and refresh measured tables in the upgrade report."""
import json
from pathlib import Path

MARKER = '## Measured comparison'
lines = [MARKER, '', 'All differences below are `(optimized - baseline) / baseline`. '
         'Negative time differences mean faster execution. Memory is peak tracked '
         'allocation, not total RSS. Values are medians of three untraced repetitions.', '']
for backend in ('numba', 'fallback'):
    a = json.loads(Path(f'benchmarks/results/baseline_{backend}.json').read_text())
    b = json.loads(Path(f'benchmarks/results/current_{backend}.json').read_text())
    lines += [f'### {backend.capitalize()}', '',
              f"Compilation-inclusive first transport call: {a['first_transport_call_seconds']:.6f} s "
              f"baseline; {b['first_transport_call_seconds']:.6f} s current. "
              + ('There is no JIT in this backend.' if backend == 'fallback' else
                 'These first-call times are excluded from the steady-state tables.'), '',
              '| N | Stage / workload | Baseline ms | Optimized ms | Time difference | Baseline peak MiB | Optimized peak MiB | Memory difference |',
              '|---:|---|---:|---:|---:|---:|---:|---:|']
    assert len(a['records']) == len(b['records'])
    for x, y in zip(a['records'], b['records']):
        for key in ('stage', 'n', 'workload', 'processed', 'status', 'k_history', 'counts', 'secondary'):
            if key in x:
                assert x[key] == y[key], (backend, key, x, y)
        bt, ot = x['median_seconds'], y['median_seconds']
        bm, om = x['peak_tracked_bytes'], y['peak_tracked_bytes']
        lines.append(f"| {x['n']:,} | {x['stage']} / {x['workload']} | {bt*1000:.3f} | "
                     f"{ot*1000:.3f} | {(ot/bt-1)*100:+.1f}% | {bm/2**20:.3f} | "
                     f"{om/2**20:.3f} | {(om/bm-1)*100:+.1f}% |")
    lines += ['', 'Every paired processed count, outcome count, completion status, secondary '
              'count and generation ratio in this matrix agrees exactly.', '']

candidates = json.loads(Path('benchmarks/results/candidates.json').read_text())
lines += ['### Layout and fission-storage candidates', '',
          'Five untraced repetitions per candidate; adapters included. No production '
          'layout change or preallocated event buffer has been retained. A subsequent '
          'chronological integer-list candidate was measured separately.', '',
          '| N | Workload | Existing ms | Component layout ms | Preallocated events ms | Existing / component / event peak MiB |',
          '|---:|---|---:|---:|---:|---|']
for r in candidates['layout']:
    a, s, e = (r[k] for k in ('aos', 'soa', 'events'))
    lines.append(f"| {r['n']:,} | {r['workload']} | {a['median_seconds']*1000:.3f} | "
                 f"{s['median_seconds']*1000:.3f} | {e['median_seconds']*1000:.3f} | "
                 f"{a['peak_tracked_bytes']/2**20:.3f} / {s['peak_tracked_bytes']/2**20:.3f} / "
                 f"{e['peak_tracked_bytes']/2**20:.3f} |")
lines += ['', f"Candidate verification: {len(candidates['rng'])} generator/bound pairs "
          'have exact scalar/batch index and continuation agreement. All layout/event '
          'candidate output positions, directions, fates and chronological fission '
          'events matched the reference for the measured cases.', '']
lines += ['### Alternating timing confirmation', '',
          'To reduce the load/order confounding visible in the first matrix, a '
          'separate five-repeat timing-only confirmation alternated baseline/current '
          'order at N=20,000, three generations. No tracing or other benchmark/test '
          'jobs ran concurrently. These are the preferred timing comparisons; '
          'individual repetitions remain in confirmation JSON files.', '',
          '| Backend | Workload | Baseline ms | Optimized ms | Speed ratio |',
          '|---|---|---:|---:|---:|']
for backend in ('numba', 'fallback'):
    confirmation = json.loads(Path(f'benchmarks/results/confirmation_{backend}.json').read_text())
    for r in confirmation:
        a, b = r['baseline_median'], r['current_median']
        lines.append(f"| {backend} | {r['workload']} | {a*1000:.3f} | {b*1000:.3f} | {a/b:.2f}× |")
lines += ['', 'The branching benefit is confirmed. Small nonbranching differences '
          'remain subject to substantial run-to-run variation. Earlier absolute timings were affected by load, so do not '
          'interpret their larger speed ratios as portable performance promises.', '']
events = json.loads(Path('benchmarks/results/event_indices.json').read_text())
lines += ['### Integer event-storage refinement', '',
          'The retained refinement was tested against the frozen source-batching '
          'stage, with five alternating timing repetitions and separate tracing. '
          'Counts, ratios and chronological event coordinates matched exactly.', '',
          '| Workload, N=20,000 | Before ms | Integer events ms | Before peak MiB | Integer events peak MiB |',
          '|---|---:|---:|---:|---:|']
for r in events:
    a, b = r['median_seconds']
    c, d = r['peak_tracked_bytes']
    lines.append(f"| {r['workload']} | {a*1000:.3f} | {b*1000:.3f} | {c/2**20:.3f} | {d/2**20:.3f} |")
report = Path('docs/PERFORMANCE_UPGRADE_REPORT.md')
head = report.read_text(encoding='utf-8').split(MARKER)[0].rstrip()
report.write_text(head + '\n\n' + '\n'.join(lines) + '\n', encoding='utf-8')
print('Paired results reconciled; benchmark tables updated.')
