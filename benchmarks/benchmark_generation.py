"""Synthetic transport benchmark; uses original physics from a Git revision.

Run from the project root: python -m benchmarks.benchmark_generation [--revision COMMIT]
JIT warmup excluded. tracemalloc includes Python and NumPy tracked allocations.
Windows peak working set includes interpreter/JIT and is process lifetime peak.
"""
import argparse
import ctypes
import gc
import json
import subprocess
import sys
import time
import tracemalloc
import types

import numpy as np
from numba_compat import njit

@njit
def seed_transport(seed):
    np.random.seed(seed)


def measure(mode, revision, n=4000, steps=60):
    if mode == 'before':
        source = subprocess.check_output(['git', 'show', f'{revision}:physics.py']).decode('utf-8')
        module = types.ModuleType('baseline_physics')
        exec(compile(source, '<git baseline>', 'exec'), module.__dict__)
    else:
        import physics as module
    positions = np.zeros((n, 3))
    directions = np.tile([0., 0., 1.], (n, 1))
    def run(p, d):
        seed_transport(2026)
        args = (p.copy(), d.copy(), 1., 0., 0., 1., 0., 1e9)
        if mode == 'before':
            return module.transport_batch(*args, max_steps=steps)
        return module.transport_batch(*args, max_steps=steps,
                                      max_samples=40 if mode == 'visualization' else 0, seed=2026)
    run(positions[:2], directions[:2])
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    result = run(positions, directions)
    elapsed = time.perf_counter() - start
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = dict(mode=mode, n=n, steps=steps, seconds=elapsed,
                  peak_tracked_mib=peak/2**20, retained_tracked_mib=retained/2**20,
                  recorded_trajectories=len(result[4]),
                  outcome_counts=np.bincount(result[2]+1, minlength=4).tolist(),
                  position_sum=result[0].sum(axis=0).tolist())
    if sys.platform == 'win32':
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
                ('peak_ws', 'ws', 'peak_pool_paged', 'pool_paged', 'peak_pool_nonpaged', 'pool_nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        record['process_peak_working_set_mib'] = counters.peak_ws/2**20
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--mode', choices=['before', 'visualization', 'performance'])
    parser.add_argument('--large', action='store_true')
    args = parser.parse_args()
    if args.mode:
        print(json.dumps(measure(args.mode, args.revision, n=100000 if args.large else 4000)))
    else:
        records = []
        for mode in ('before', 'visualization', 'performance'):
            output = subprocess.check_output([sys.executable, __file__, '--mode', mode, '--revision', args.revision])
            records.append(json.loads(output))
        assert all(r['outcome_counts'] == records[0]['outcome_counts'] for r in records)
        assert all(np.allclose(r['position_sum'], records[0]['position_sum']) for r in records)
        print(json.dumps(records, indent=2))
