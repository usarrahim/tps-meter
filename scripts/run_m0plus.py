"""M0+ re-simulation check and trust v3 (v2 + M0+) for the frozen M2 v1 ensemble.

For every measurement of the evaluated splits, one worker process
  1. rebuilds the preprocessed record exactly as in Phase 6 (same effect seeds),
  2. runs the M2 ensemble (ONNX, normalisation inside) and the classic fit M0,
  3. re-simulates the measurement at the M2 estimate (eval.m0plus.m0plus_light) -> chi2_nu.
Trust v3 thresholds (shared false-alarm budget, configs/trust.yaml) come from val only. Test splits:
THIRD USE (Phase 6 first, trust v2 second); nothing here is tuned on them.
Optionally the full M0+ inversion (eval.m0plus.fit_m0plus) runs on a small val subset of high-k
measurements to show what the fitted contacts and k look like (development check, val only).

Resumable: results are saved every chunk; re-running the same command continues.
The film of each synthetic measurement is its own film (the "calibrated sensor" assumption, M0PLUS.md).

Usage:
  python scripts/run_m0plus.py --onnx models/m2_v1/m2_ensemble.onnx --workers 40
  python scripts/run_m0plus.py --onnx ... --split-files val=pilot.h5 --splits val --limit 200   (dry run)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from tpslib.config import load_yaml
from tpslib.data.dataset import TPSDataset, contact_asymmetry, effect_seed
from tpslib.data.generate import dataset_dir
from tpslib.data.prior import FAULTS
from tpslib.eval.m0plus import load_m0plus_config
from tpslib.eval.metrics import summarize
from tpslib.eval.report import INK2, SERIES, plt, save, style
from tpslib.eval.trust import (TARGET_NAMES, fit_thresholds_v2, flags_v2, load_trust_config)
from tpslib.runtime import runs_root

FIELDS = ("mean", "std", "std_epistemic", "m0_k", "m0_resid", "dT_max", "chi2", "chi2_by_split", "best_split",
          "sim_s")
THIRD_USE = ("THIRD USE of the test splits (Phase 6 first, trust v2 second). The M0+ check and the trust v3 "
             "thresholds were chosen on val only; nothing was tuned on the test splits.")
FAULT_LABELS = {"air_gap_low_hc": "air gap", "thin_sample": "thin sample", "drift_jump": "drift jump",
                "contact_asymmetry_gt_max": "contact asymmetry > 10"}

_W: dict = {}


def _init(path, base_seed, onnx_path):
    import onnxruntime as ort
    from tpslib.config import load_sensor
    from tpslib.sim.sensor_model import load_numerics, load_sensor_raw
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _W.update(ds=TPSDataset(path, train=False, base_seed=base_seed),
              sess=ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"]),
              m=load_sensor().n_rings, cfg=load_m0plus_config(), sc=load_sensor_raw(), nu=load_numerics())


def _work(i):
    """All per-measurement results of item i of the split file."""
    from tpslib.data.features import make_features
    from tpslib.eval.m0plus import film_from_row, m0plus_light
    from tpslib.fit_tps import fit_tps
    ds = _W["ds"]
    proc, prm = ds.processed(int(i))
    f = make_features(proc)
    mean, std, mem = _W["sess"].run(None, {"seq": f["seq"][None], "scalars": f["scalars"][None]})
    mean, std, epi = mean[0], std[0], mem[:, 0, :].std(axis=0)
    try:
        r0 = fit_tps(proc.t, proc.dT, proc.P, proc.a, _W["m"])
        m0k, m0r = r0.k, r0.residual_rms
    except Exception:  # noqa: BLE001 - a failed classic fit is itself a result
        m0k, m0r = np.nan, np.nan
    start = {"k": float(np.exp(mean[0])), "alpha": float(np.exp(mean[1])), "G_c": float(np.exp(mean[2]))}
    lt = m0plus_light(proc, start, film_from_row(prm), _W["cfg"], _W["sc"], _W["nu"])
    return int(i), {"mean": mean, "std": std, "std_epistemic": epi, "m0_k": m0k, "m0_resid": m0r,
                    "dT_max": float(np.mean(proc.dT[-50:])), "chi2": lt["chi2_nu"],
                    "chi2_by_split": lt["chi2_by_split"], "best_split": lt["best_split"], "sim_s": lt["time_s"]}


def run_split(path: Path, base_seed: int, idx: np.ndarray, onnx: Path, workers: int, store: Path, log) -> dict:
    """Results for the items `idx` of one split file; resumes from `store` (npz)."""
    res = {}
    if store.exists():
        z = np.load(store)
        for j, i in enumerate(z["index"]):
            res[int(i)] = {k: z[k][j] for k in FIELDS}
    todo = [int(i) for i in idx if int(i) not in res]
    log(f"{path.name}: {len(res)} done, {len(todo)} to go")

    def dump():
        keys = sorted(res)
        np.savez(store, index=np.array(keys), **{k: np.array([res[i][k] for i in keys]) for k in FIELDS})

    if todo:
        t0 = time.perf_counter()
        with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(path, base_seed, onnx)) as pool:
            for n, (i, r) in enumerate(pool.imap_unordered(_work, todo, chunksize=2), 1):
                res[i] = r
                if n % 200 == 0 or n == len(todo):
                    dump()
                    rate = n / (time.perf_counter() - t0)
                    log(f"  {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:.0f} min")
    keys = np.array([int(i) for i in idx])
    return {"index": keys, **{k: np.array([res[int(i)][k] for i in keys]) for k in FIELDS}}


def scores_of(r: dict) -> dict:
    """Trust scores (same definitions as eval.trust.trust_scores) plus the M0+ chi2."""
    with np.errstate(invalid="ignore", divide="ignore"):
        s = {"total_std": r["std"][:, 0], "m0_m2": np.abs(np.log(r["m0_k"]) - r["mean"][:, 0]),
             "m0_resid": r["m0_resid"] / np.abs(r["dT_max"]), "dT_max": r["dT_max"],
             "m0plus_chi2": np.log(r["chi2"])}
    for i, t in enumerate(TARGET_NAMES):
        s[f"pred_{t}"] = r["mean"][:, i]
        s[f"epistemic_{t}"] = r["std_epistemic"][:, i]
    return s


def sub(s: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in s.items()}


def main() -> None:
    """Run M0+ light on the splits, fit trust v3 on val, report detection and false alarms."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="m2_ensemble.onnx (normalisation inside)")
    ap.add_argument("--splits", nargs="+", default=["val", "test", "test_faulty"])
    ap.add_argument("--split-files", nargs="*", default=None, help="override, e.g. val=pilot.h5 (dry runs)")
    ap.add_argument("--limit", type=int, default=None, help="first N items per split (dry runs)")
    ap.add_argument("--limits", nargs="*", default=None,
                    help="per-split item counts, e.g. test=2500 (the first N items; splits are i.i.d.)")
    ap.add_argument("--max-asym", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=None, help="default: <runs root>/../reports/m0plus")
    args = ap.parse_args()
    T0 = time.perf_counter()
    log = lambda m: print(f"[{time.perf_counter() - T0:6.0f}s] {m}", flush=True)
    out = Path(args.out) if args.out else Path(runs_root()).parent / "reports" / "m0plus"
    out.mkdir(parents=True, exist_ok=True)
    tcfg = load_trust_config()
    data_dir = dataset_dir(load_yaml("configs/dataset_v1.yaml"))
    files = {s: f"{s}.h5" for s in args.splits}
    for kv in args.split_files or []:
        k_, v_ = kv.split("="); files[k_] = v_

    R, P = {}, {}
    for split in args.splits:
        path = data_dir / files[split]
        ds = TPSDataset(path, train=False, base_seed=effect_seed(split))
        lim = dict(kv.split("=") for kv in args.limits or []).get(split, args.limit)
        idx = np.arange(len(ds)) if lim is None else np.arange(min(int(lim), len(ds)))
        R[split] = run_split(path, effect_seed(split), idx, Path(args.onnx), args.workers,
                             out / f"results_{split}{'_dry' if args.split_files else ''}.npz", log)
        P[split] = {k: v[idx] for k, v in ds.params.items()}

    S = {s: scores_of(R[s]) for s in R}
    dom = {s: contact_asymmetry(P[s]) <= args.max_asym for s in R}
    dT_range = tcfg["dT_range_K"]
    sv = sub(S["val"], dom["val"])
    thr = {v: fit_thresholds_v2(sv, tcfg, version=v) for v in ("v2", "v3")}
    summary = {"created": dt.datetime.now().isoformat(timespec="seconds"), "onnx": str(args.onnx),
               "splits": {s: int(len(R[s]["index"])) for s in R}, "dry_run": bool(args.split_files),
               "test_use": THIRD_USE, "thresholds": thr,
               "sim_time_mean_s": float(np.nanmean(np.concatenate([R[s]["sim_s"] for s in R]))),
               "chi2_failed": {s: int(np.isnan(R[s]["chi2"]).sum()) for s in R}}

    # chi2 by k band and contact domain (val: development view)
    rows = []
    for s in R:
        k = P[s]["k"]
        for name, m in (("asym<=10", dom[s]), ("asym>10", ~dom[s])):
            for lo, hi in ((0.05, 0.5), (0.5, 5), (5, 30.01)):
                sel = m & (k >= lo) & (k < hi)
                if sel.any():
                    c = R[s]["chi2"][sel]
                    rows.append({"split": s, "domain": name, "k_band": f"{lo}-{min(hi, 30)}", "n": int(sel.sum()),
                                 "chi2_median": float(np.nanmedian(c)), "chi2_p95": float(np.nanquantile(c, 0.95))})
    summary["chi2_by_band"] = rows

    # detection: v2 vs v3 (thresholds from val)
    det = {}
    for ver in ("v2", "v3"):
        fl = lambda s: flags_v2(S[s], thr[ver], dT_range)
        d = {"false_alarm_val_in_domain": float(fl("val")["any"][dom["val"]].mean())}
        d["val_asym_gt_max"] = float(fl("val")["any"][~dom["val"]].mean())
        if "test" in R:
            ft = fl("test")
            d["false_alarm_test_in_domain"] = float(ft["any"][dom["test"]].mean())
            d["contact_asymmetry_gt_max"] = float(ft["any"][~dom["test"]].mean())
            bad = ~dom["test"] & ~ft["any"]
            err = np.exp(R["test"]["mean"][:, 0]) / P["test"]["k"] - 1
            d["asym_unflagged_k_err"] = summarize(err[bad]) if bad.any() else None
            hi_k = ~dom["test"] & (P["test"]["k"] >= 5)
            d["contact_asymmetry_gt_max_k_ge_5"] = float(ft["any"][hi_k].mean()) if hi_k.any() else None
        if "test_faulty" in R:
            ff = fl("test_faulty")
            kind = P["test_faulty"]["fault"]
            for fname, code in FAULTS.items():
                if code and (kind == code).any():
                    d[fname] = float(ff["any"][kind == code].mean())
            err = np.exp(R["test_faulty"]["mean"][:, 0]) / P["test_faulty"]["k"] - 1
            d["faulty_unflagged_k_err"] = summarize(err[~ff["any"]]) if (~ff["any"]).any() else None
        if ver == "v3":
            d["by_m0plus_chi2_test_faulty"] = float(ff["m0plus_chi2"].mean()) if "test_faulty" in R else None
        det[ver] = d
    summary["detection"] = det
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out / "trust_v3.json").write_text(json.dumps({"fitted_on": "val (contact asymmetry <= max_asym)",
                                                   "thresholds": thr["v3"], "dT_range_K": dT_range,
                                                   "config": tcfg["v3"], "m0plus": load_m0plus_config()},
                                                  indent=2, default=float), encoding="utf-8")

    # figures
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    bins = np.geomspace(0.05, 1e4, 60)
    name = lambda s: Path(files[s]).stem          # the file actually used (differs in dry runs)
    groups = [(f"{name('val')}, asym <= 10", R["val"]["chi2"][dom["val"]], SERIES[0]),
              (f"{name('val')}, asym > 10", R["val"]["chi2"][~dom["val"]], SERIES[1])]
    if "test_faulty" in R:
        kind = P["test_faulty"]["fault"]
        for j, (fname, lab) in enumerate((("air_gap_low_hc", "air gap"), ("thin_sample", "thin sample"),
                                          ("drift_jump", "drift jump"))):
            groups.append((f"{name('test_faulty')}: {lab}", R["test_faulty"]["chi2"][kind == FAULTS[fname]],
                           SERIES[2 + j]))
    for lab, c, col in groups:
        c = np.clip(c[np.isfinite(c)], bins[0], bins[-1])
        ax.hist(c, bins=bins, histtype="step", lw=1.8, color=col, label=f"{lab} (n={c.size})",
                weights=np.full(c.size, 1.0 / max(c.size, 1)))
    ax.axvline(np.exp(thr["v3"]["upper"]["m0plus_chi2"]), color=INK2, ls="--", lw=1, label="v3 threshold")
    ax.set_xscale("log"); ax.set_xlabel("M0+ chi2_nu at the M2 estimate"); ax.set_ylabel("fraction per bin")
    ax.legend(frameon=False, fontsize=8); style(ax, "Re-simulation residual (threshold from val)")
    save(fig, out / "m0plus_chi2.png")
    if "test_faulty" in R or "test" in R:
        names = [n for n in ("air_gap_low_hc", "thin_sample", "drift_jump", "contact_asymmetry_gt_max") if n in det["v3"]]
        x = np.arange(len(names) + 1)
        fig, ax = plt.subplots(figsize=(9, 4.4))
        for j, ver in enumerate(("v2", "v3")):
            vals = [100 * det[ver][n] for n in names] + [100 * det[ver].get("false_alarm_test_in_domain", np.nan)]
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=SERIES[j], label=f"trust {ver}")
        ax.set_xticks(x, [FAULT_LABELS[n] for n in names] + ["false alarm (normal)"], fontsize=8)
        ax.axhline(90, color=INK2, ls="--", lw=1.0, label="plan target 90%")
        ax.set_ylabel("flagged [%]"); ax.set_ylim(0, 105); ax.legend(frameon=False)
        style(ax, "Flagging faulty measurements: v2 vs v3 with M0+ (thresholds from val; test third use)")
        save(fig, out / "faulty_detection_v3.png")

    shutil.make_archive(str(out.parent.parent / "m0plus_bundle"), "zip", out)
    log(f"done -> {out}")
    print(json.dumps({"detection": det, "chi2_by_band": rows, "chi2_failed": summary["chi2_failed"],
                      "sim_time_mean_s": summary["sim_time_mean_s"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
