"""Measure simulation throughput for several worker counts and suggest the best one.

Parallel generation is limited by memory bandwidth, so more processes than cores is not always
faster. Run this once on a new machine (e.g. a Colab runtime) before the full generation.

Usage: python scripts/bench_generation.py [--samples 48] [--workers 4 8 16 ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from tpslib.config import load_yaml
from tpslib.data.generate import _worker
from tpslib.data.prior import row, sample_split


def main() -> None:
    """Benchmark the given worker counts and print the fastest."""
    ncpu = os.cpu_count() or 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_v1.yaml")
    ap.add_argument("--samples", type=int, default=None, help="samples per setting (default 3 x cpu count)")
    ap.add_argument("--workers", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    candidates = args.workers or sorted({max(1, ncpu // 4), max(1, ncpu // 2), max(1, 3 * ncpu // 4), ncpu})
    n = args.samples or 3 * ncpu
    p = sample_split(1000, int(cfg["pilot"]["seed"]))   # pilot parameters: representative mix
    tasks = [(i, row(p, i)) for i in range(n)]
    total = sum(int(s["n"]) for s in cfg["splits"].values())
    results = []
    print(f"cpu_count={ncpu}, {n} samples per setting, full dataset = {total} samples", flush=True)
    for w in candidates:
        with mp.get_context("spawn").Pool(w) as pool:
            pool.map(_worker, tasks[:w])        # warm-up: imports and D(tau) table
            t0 = time.perf_counter()
            res = pool.map(_worker, tasks, chunksize=1)
            wall = time.perf_counter() - t0
        rate = n / wall
        results.append({"workers": w, "samples_per_s": rate,
                        "per_sim_s": float(np.mean([r[1]["sim_time_s"] for r in res])),
                        "full_run_h": total / rate / 3600})
        print(json.dumps(results[-1]), flush=True)
    best = max(results, key=lambda r: r["samples_per_s"])
    print(f"\nBEST: --workers {best['workers']}  (~{best['full_run_h']:.1f} h for {total} samples)")


if __name__ == "__main__":
    main()
