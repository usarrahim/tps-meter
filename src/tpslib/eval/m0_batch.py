"""Run the classic TPS fit (M0) on every item of a dataset, in parallel.

Each worker rebuilds the same preprocessed record the model sees (same seed, same effects), so M0 and
M2 are compared on identical measurements.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

_DS = None
_M = None


def _init(path, base_seed, indices):
    global _DS, _M
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from tpslib.data.dataset import TPSDataset
    from tpslib.config import load_sensor
    _DS = TPSDataset(path, train=False, base_seed=base_seed, indices=indices)
    _M = load_sensor().n_rings


def _fit(j):
    from tpslib.fit_tps import fit_tps
    proc, prm = _DS.processed(j)
    try:
        r = fit_tps(proc.t, proc.dT, proc.P, proc.a, _M)
        return j, r.k, r.alpha, r.residual_rms, float(np.mean(proc.dT[-50:])), 1
    except Exception:  # noqa: BLE001 - a failed classic fit is itself a result
        return j, np.nan, np.nan, np.nan, float(np.mean(proc.dT[-50:])), 0


def run_m0(ds, workers: int | None = None) -> dict:
    """M0 results for all items of `ds` (eval mode). Returns arrays aligned with ds.indices."""
    n = len(ds)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    out = np.full((n, 4), np.nan)
    ok = np.zeros(n, bool)
    with mp.get_context("spawn").Pool(workers, initializer=_init,
                                      initargs=(ds.path, ds.base_seed, ds.indices)) as pool:
        for j, k, a, res, dtmax, good in pool.imap_unordered(_fit, range(n), chunksize=16):
            out[j] = (k, a, res, dtmax)
            ok[j] = bool(good)
    return {"k": out[:, 0], "alpha": out[:, 1], "residual_rms": out[:, 2], "dT_max": out[:, 3], "ok": ok}
