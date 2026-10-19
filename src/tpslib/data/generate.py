"""Parallel, resumable dataset generation to HDF5 shards, merged per split, with a manifest.

Layout (under <data root>/<output_dir>):
  <split>_shards/shard_00000.h5 ...   written atomically (tmp file + rename) when a shard is full
  <split>.h5                          merged file, created when all shards exist; shards are then removed
  manifest.json                       configs (+ sha256), seeds, git commit, code hash, counts, timings

Optional mirror directory (e.g. Google Drive on Colab): every finished shard and the merged file are
copied there, and existing ones are restored from it on start, so generation survives a lost session
while the work itself runs on the fast local disk.

Every sample stores all parameters, the clean dT(t) and P(t) on a log-time grid, and sim diagnostics.
Measurement effects (noise, drift, ADC, calibration errors) are NOT applied here (PROJECT_RULES.md data rules).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.data.prior import FILM_LAYERS, row, sample_split
from tpslib.runtime import PROJECT_ROOT, data_root

_SIM_CACHE: dict = {}


def _sim_configs():
    if not _SIM_CACHE:
        from tpslib.sim.sensor_model import load_numerics, load_sensor_raw
        _SIM_CACHE["sensor"] = load_sensor_raw()
        _SIM_CACHE["numerics"] = load_numerics()
    return _SIM_CACHE["sensor"], _SIM_CACHE["numerics"]


def simulate_params(prm: dict) -> dict:
    """Run one simulation for a parameter row. Never raises; failures are flagged."""
    from tpslib.sim.sensor_model import SimParams, simulate

    sensor_cfg, numerics = _sim_configs()
    n_out = int(numerics["output"]["n_points"])
    t0 = time.perf_counter()
    try:
        sp = SimParams(
            k=prm["k"], rho_cp=prm["rho_cp"], P0=prm["P0"], t_end=prm["t_end"],
            h_c_bottom=prm["h_c_bottom"], h_c_top=prm["h_c_top"],
            sample_radius=prm["sample_radius"],
            thickness_bottom=prm["sample_thickness"], thickness_top=prm["sample_thickness"],
            k_polyimide=prm["k_polyimide"], k_adhesive=prm["k_adhesive"],
            layer_scale={name: prm[f"scale_{name}"] for name in FILM_LAYERS},
            constant_current=True, t_out_min=1.0 / prm["sps"],
        )
        r = simulate(sp, sensor_cfg=sensor_cfg, numerics=numerics)
        return {"t": r.t, "dT": r.dT, "P": r.P, "ok": 1, "error": "",
                "sim_time_s": time.perf_counter() - t0,
                "energy_rel_imbalance": r.energy["rel_imbalance"],
                "frac_lost": r.energy["frac_lost"], "n_steps": r.info["n_steps"]}
    except Exception as exc:  # noqa: BLE001 - keep the run alive, record the failure
        nan = np.full(n_out, np.nan)
        return {"t": nan, "dT": nan, "P": nan, "ok": 0, "error": repr(exc)[:200],
                "sim_time_s": time.perf_counter() - t0, "energy_rel_imbalance": np.nan,
                "frac_lost": np.nan, "n_steps": 0}


def _worker(args):
    i, prm = args
    return i, simulate_params(prm)


def _write_h5(path: Path, params: dict, results: list[dict], attrs: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        g = f.create_group("params")
        for k, v in params.items():
            g.create_dataset(k, data=v)
        f.create_dataset("t", data=np.stack([r["t"] for r in results]).astype(np.float64))
        f.create_dataset("dT", data=np.stack([r["dT"] for r in results]).astype(np.float32))
        f.create_dataset("P", data=np.stack([r["P"] for r in results]).astype(np.float32))
        d = f.create_group("diag")
        for key in ("ok", "sim_time_s", "energy_rel_imbalance", "frac_lost", "n_steps"):
            d.create_dataset(key, data=np.array([r[key] for r in results]))
        d.create_dataset("error", data=np.array([r["error"] for r in results], dtype=h5py.string_dtype()))
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["complete"] = True
    os.replace(tmp, path)


def _shard_complete(path: Path, n: int, code: str | None = None) -> bool:
    """Shard exists, is complete, has n samples and (if given) was made by the same code."""
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            ok = bool(f.attrs.get("complete", False)) and f["dT"].shape[0] == n
            if ok and code is not None and f.attrs.get("code_hash") != code:
                raise RuntimeError(f"{path} was made by different code (code_hash mismatch); "
                                   "delete it or restore the matching code")
            return ok
    except OSError:
        return False


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def merge_shards(shard_paths: list[Path], out: Path, attrs: dict) -> None:
    """Concatenate shard files into one HDF5 file (written atomically)."""
    tmp = out.with_suffix(".tmp")
    with h5py.File(tmp, "w") as fo:
        first = True
        for sp in shard_paths:
            with h5py.File(sp, "r") as fi:
                def copy(name, obj):
                    if not isinstance(obj, h5py.Dataset):
                        return
                    data = obj[()]
                    if first:
                        maxshape = (None,) + data.shape[1:]
                        fo.create_dataset(name, data=data, maxshape=maxshape, chunks=True)
                    else:
                        ds = fo[name]
                        n0 = ds.shape[0]
                        ds.resize(n0 + data.shape[0], axis=0)
                        ds[n0:] = data
                fi.visititems(copy)
            first = False
        for k, v in attrs.items():
            fo.attrs[k] = v
        fo.attrs["complete"] = True
    os.replace(tmp, out)


def git_info() -> dict:
    """Current git commit and whether src/configs/scripts have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "src", "configs", "scripts"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True)
        return {"commit": sha.stdout.strip() or None, "dirty": bool(dirty.stdout.strip())}
    except OSError:
        return {"commit": None, "dirty": None}


def code_hash() -> str:
    """sha256 over src/ and configs/ (reproducibility even without a git commit)."""
    h = hashlib.sha256()
    for root in ("src", "configs"):
        for p in sorted((PROJECT_ROOT / root).rglob("*")):
            if p.is_file() and p.suffix in (".py", ".yaml"):
                h.update(p.relative_to(PROJECT_ROOT).as_posix().encode())
                h.update(p.read_bytes())
    return h.hexdigest()


def _config_snapshot(names) -> dict:
    out = {}
    for name in names:
        p = CONFIG_DIR / f"{name}.yaml"
        out[name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "content": load_yaml(p)}
    return out


def run_split(name: str, n: int, seed: int, out_dir: Path, workers: int, shard_size: int,
              k_band=None, faulty: bool = False, progress: bool = True, mirror: Path | None = None) -> dict:
    """Generate one split (resumable). Returns split statistics for the manifest.

    mirror: optional directory (e.g. on Google Drive) that receives a copy of every finished shard
    and of the merged file, and from which they are restored on start.
    """
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{name}.h5"
    params = sample_split(n, seed, k_band=k_band, faulty=faulty)
    code = code_hash()
    if mirror is not None and not final.exists() and (mirror / final.name).exists():
        _copy_atomic(mirror / final.name, final)
    if final.exists():
        with h5py.File(final, "r") as f:
            if f.attrs.get("complete") and f["dT"].shape[0] == n:
                saved = f["params/k"][()]
                if not np.allclose(saved, params["k"]):
                    raise RuntimeError(f"{final} exists but was made with different parameters")
                return json.loads(f.attrs["stats"])
    shard_dir = out_dir / f"{name}_shards"
    shard_dir.mkdir(exist_ok=True)
    n_shards = int(np.ceil(n / shard_size))
    shard_paths = [shard_dir / f"shard_{s:05d}.h5" for s in range(n_shards)]
    bounds = [(s * shard_size, min(n, (s + 1) * shard_size)) for s in range(n_shards)]
    if mirror is not None:
        for sp_, (lo, hi) in zip(shard_paths, bounds):
            mp_ = mirror / shard_dir.name / sp_.name
            if not _shard_complete(sp_, hi - lo, code) and _shard_complete(mp_, hi - lo, code):
                _copy_atomic(mp_, sp_)
    todo = [s for s in range(n_shards) if not _shard_complete(shard_paths[s], bounds[s][1] - bounds[s][0], code)]
    attrs = {"split": name, "seed": seed, "n_total": n, "code_hash": code}

    t_start = time.perf_counter()
    tasks = [(i, row(params, i)) for s in todo for i in range(*bounds[s])]
    buffers: dict[int, dict] = {s: {} for s in todo}
    shard_of = {i: s for s in todo for i in range(*bounds[s])}
    if tasks:
        ctx = mp.get_context("spawn")
        bar = tqdm(total=len(tasks), desc=name, disable=not progress, smoothing=0.05)
        with ctx.Pool(workers) as pool:
            for i, res in pool.imap(_worker, tasks, chunksize=1):
                s = shard_of[i]
                buffers[s][i] = res
                lo, hi = bounds[s]
                if len(buffers[s]) == hi - lo:
                    sl = {k: v[lo:hi] for k, v in params.items()}
                    _write_h5(shard_paths[s], sl, [buffers[s][j] for j in range(lo, hi)],
                              {**attrs, "shard": s, "lo": lo, "hi": hi})
                    if mirror is not None:
                        _copy_atomic(shard_paths[s], mirror / shard_dir.name / shard_paths[s].name)
                    del buffers[s]
                bar.update(1)
        bar.close()
    wall = time.perf_counter() - t_start

    # statistics over all shards
    ok, tsim = [], []
    for sp in shard_paths:
        with h5py.File(sp, "r") as f:
            ok.append(f["diag/ok"][()]); tsim.append(f["diag/sim_time_s"][()])
    ok, tsim = np.concatenate(ok), np.concatenate(tsim)
    stats = {"n": n, "seed": seed, "n_ok": int(ok.sum()), "n_failed": int((ok == 0).sum()),
             "wall_time_s_this_run": wall, "sim_time_mean_s": float(tsim.mean()),
             "sim_time_p95_s": float(np.percentile(tsim, 95)), "sim_time_sum_s": float(tsim.sum()),
             "workers": workers, "shard_size": shard_size, "k_band": k_band, "faulty": faulty,
             "finished": _dt.datetime.now().isoformat(timespec="seconds")}
    merge_shards(shard_paths, final, {**attrs, "stats": json.dumps(stats)})
    if mirror is not None:
        _copy_atomic(final, mirror / final.name)
        shutil.rmtree(mirror / shard_dir.name, ignore_errors=True)
    for sp in shard_paths:
        sp.unlink()
    shard_dir.rmdir()
    return stats


def update_manifest(out_dir: Path, dataset_cfg: dict, split: str, stats: dict) -> None:
    """Record configs, code version and the statistics of a finished split in manifest.json."""
    path = out_dir / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    man.update({
        "dataset_version": dataset_cfg.get("version"),
        "updated": _dt.datetime.now().isoformat(timespec="seconds"),
        "git": git_info(),
        "code_hash": code_hash(),
        "configs": _config_snapshot(["sensor", "device", "sim_numerics", "prior_v1", "dataset_v1"]),
        "note": "Clean curves only; measurement effects are applied at training time (PROJECT_RULES.md).",
    })
    man.setdefault("splits", {})[split] = stats
    path.write_text(json.dumps(man, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def dataset_dir(dataset_cfg: dict) -> Path:
    """Directory of a dataset version under the data root (TPS_DATA_ROOT or ./data)."""
    return data_root() / dataset_cfg["output_dir"]
