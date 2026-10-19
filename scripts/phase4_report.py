"""Phase 4 gate figures: noisy realisations + feature channels, normalised distributions, M0 on noisy
ideal curves, per-effect sensitivity and feature timing.

Usage: python scripts/phase4_report.py [--data data/synthetic/v1/pilot.h5] [--out reports/phase4]
The normaliser here is fitted on the given file as a stand-in; the real one is fitted on train (Phase 5).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tpslib.config import load_sensor
from tpslib.data.dataset import Normalizer, TPSDataset
from tpslib.data.features import CHANNELS, SCALARS, TARGETS, make_features
from tpslib.eval.report import BAND, INK2, MARKERS, SERIES, plt, save, style
from tpslib.fit_tps import fit_tps
from tpslib.measurement.acquisition import acquire
from tpslib.measurement.preprocess import preprocess
from tpslib.runtime import PROJECT_ROOT
from tpslib.synth_analytic import clean_record_case as analytic_case

S = load_sensor()
A, M = S.outer_radius, S.n_rings
EFFECTS = ["noise", "drift", "timing", "power", "electronic", "calibration"]
CH_LABEL = {"theta_rel": "θ / θ_ref", "dtheta_dlnt_rel": "dθ/d ln t / θ_ref", "log_count": "ln(1 + örnek sayısı)"}


def realisations(ds: TPSDataset, out: Path):
    """Five noisy realisations of one clean curve and their feature channels (low and high k)."""
    picks = {}
    k = ds.params["k"]
    for name, target in [("düşük k", 0.2), ("yüksek k", 15.0)]:
        picks[name] = int(np.argmin(np.abs(np.log(k) - np.log(target))))
    fig, axs = plt.subplots(2, 4, figsize=(18, 7.5))
    f = ds._h5 = ds._h5 or __import__("h5py").File(ds.path, "r")
    for row, (name, i) in enumerate(picks.items()):
        prm = ds.row(i)
        t, dT = f["t"][i], f["dT"][i].astype(float)
        clean = make_features(preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects=False)))
        ax = axs[row, 0]
        for r in range(5):
            p = preprocess(acquire(t, dT, prm, np.random.default_rng(100 + r)))
            F = make_features(p)
            col = SERIES[r]
            ax.plot(p.t, p.dT, color=col, lw=0.6, alpha=0.8)
            for c in range(3):
                axs[row, c + 1].plot(F["t_grid"], F["seq"][c], color=col, lw=1.0, alpha=0.9,
                                     label=f"örnek {r + 1}" if c == 0 else None)
        ax.plot(t, dT, color="black", lw=1.2, ls="--", label="temiz")
        ax.set_xscale("log"); ax.set_xlabel("t [s]"); ax.set_ylabel("ΔT [K]")
        ax.legend(frameon=False, fontsize=8)
        style(ax, f"{name}: k={prm['k']:.3g} W/m·K, {prm['sps']:.0f} SPS, t_son={prm['t_end']:.0f} s")
        for c in range(3):
            a2 = axs[row, c + 1]
            a2.plot(clean["t_grid"], clean["seq"][c], color="black", lw=1.0, ls="--")
            a2.set_xscale("log"); a2.set_xlabel("t [s] (256 log bin)")
            style(a2, CH_LABEL[CHANNELS[c]])
        axs[row, 1].legend(frameon=False, fontsize=7)
    save(fig, out / "realisations_and_channels.png")
    return picks


def normalised(ds: TPSDataset, out: Path) -> Normalizer:
    """Fit the normaliser on the given file and plot the normalised distributions."""
    nm = Normalizer.fit(ds, n=len(ds), seed=0)
    nm.save(out / "normalizer_pilot.json")
    seqs, scals, ys = [], [], []
    for j in range(len(ds)):
        f, y = ds.features(j)
        s, c, yy = nm.apply(f["seq"], f["scalars"], y)
        seqs.append(s); scals.append(c); ys.append(yy)
    S_, C_, Y_ = np.stack(seqs), np.stack(scals), np.stack(ys)
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    for c in range(S_.shape[1]):
        axs[0].hist(S_[:, c, :].ravel(), bins=120, histtype="step", lw=1.8, color=SERIES[c],
                    range=(-5, 5), density=True, label=CH_LABEL[CHANNELS[c]])
    axs[0].legend(frameon=False, fontsize=8); axs[0].set_xlabel("normalize değer"); axs[0].set_ylabel("yoğunluk")
    style(axs[0], "Dizi kanalları (tüm binler)")
    axs[1].boxplot([C_[:, i] for i in range(C_.shape[1])], tick_labels=SCALARS, vert=True, showfliers=True,
                   flierprops={"markersize": 2})
    axs[1].tick_params(axis="x", rotation=40)
    style(axs[1], "Skaler girdiler")
    axs[2].boxplot([Y_[:, i] for i in range(Y_.shape[1])], tick_labels=TARGETS, flierprops={"markersize": 2})
    style(axs[2], "Hedefler")
    save(fig, out / "normalised_distributions.png")
    stats = {"seq_channel_mean": S_.mean(axis=(0, 2)).tolist(), "seq_channel_std": S_.std(axis=(0, 2)).tolist(),
             "scalar_mean": C_.mean(axis=0).tolist(), "scalar_std": C_.std(axis=0).tolist(),
             "finite": bool(np.isfinite(S_).all() and np.isfinite(C_).all())}
    return nm, stats


def m0_gate(out: Path) -> dict:
    """Phase 4 gate: M0 on noisy ideal curves for 20 k values."""
    ks = np.geomspace(0.05, 30.0, 20)
    rows = []
    for k in ks:
        rc = float(np.interp(np.log(k), np.log([0.05, 30]), [1.0e6, 4.0e6]))
        t, dT, prm = analytic_case(float(k), rho_cp=rc, alpha_R_calib_rel=0.0, radius_calib_rel=0.0)
        for mode, eff in [("ölçüm etkileri", {e: True for e in EFFECTS if e != "calibration"}),
                          ("+ kalibrasyon hatası", True)]:
            for r in range(5):
                prm_r = dict(prm)
                if mode != "ölçüm etkileri":
                    g = np.random.default_rng(900 + r)
                    prm_r["alpha_R_calib_rel"] = g.normal(0, 0.01)
                    prm_r["radius_calib_rel"] = g.normal(0, 0.005)
                p = preprocess(acquire(t, dT, prm_r, np.random.default_rng(int(k * 1e4) + r), effects=eff))
                f = fit_tps(p.t, p.dT, p.P, p.a, M)
                rows.append({"k": k, "mode": mode, "rep": r, "err_k_pct": (f.k / k - 1) * 100,
                             "err_alpha_pct": (f.alpha / prm["alpha"] - 1) * 100})
    d = pd.DataFrame(rows)
    d.to_csv(out / "m0_noisy_gate.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, lab, lim in [(axs[0], "err_k_pct", "k", 2.0), (axs[1], "err_alpha_pct", "α", None)]:
        if lim:
            ax.axhspan(-lim, lim, color=BAND, zorder=0)
        for i, (mode, g) in enumerate(d.groupby("mode", sort=False)):
            ax.plot(g.k, g[col], MARKERS[i], color=SERIES[i], ms=5, alpha=0.8, label=mode)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log")
        ax.set_xlabel("k [W/m·K]"); ax.set_ylabel(f"M0 {lab} hatası [%]")
        ax.legend(frameon=False, fontsize=8)
        style(ax, f"M0, gürültülü ideal eğriler: {lab} (5 tekrar/k" + (", gri: ±%2 kapı)" if lim else ")"))
    save(fig, out / "m0_noisy_gate.png")
    meas = d[d["mode"] == "ölçüm etkileri"]
    return {"passed": bool((meas.err_k_pct.abs() < 2).all()),
            "max_abs_err_k_pct": float(meas.err_k_pct.abs().max()),
            "max_abs_err_alpha_pct": float(meas.err_alpha_pct.abs().max()),
            "with_calibration_max_abs_err_k_pct": float(d[d["mode"] != "ölçüm etkileri"].err_k_pct.abs().max())}


def sensitivity(out: Path) -> None:
    """M0 error caused by each measurement effect alone."""
    rows = []
    for k in [0.05, 0.19, 1.2, 16.0]:
        rc = float(np.interp(np.log(k), np.log([0.05, 30]), [1.0e6, 4.0e6]))
        t, dT, prm = analytic_case(k, rho_cp=rc, alpha_R_calib_rel=0.01, radius_calib_rel=0.005)
        for eff in EFFECTS:
            ek, ea = [], []
            for s in range(20):
                p = preprocess(acquire(t, dT, prm, np.random.default_rng(s), effects={eff: True}))
                f = fit_tps(p.t, p.dT, p.P, p.a, M)
                ek.append(f.k / k - 1); ea.append(f.alpha / prm["alpha"] - 1)
            rows.append({"k": k, "t_end_s": prm["t_end"], "sps": prm["sps"], "effect": eff,
                         "max_abs_err_k_pct": np.abs(ek).max() * 100, "max_abs_err_alpha_pct": np.abs(ea).max() * 100})
    pd.DataFrame(rows).to_csv(out / "m0_effect_sensitivity.csv", index=False)


def timing(ds: TPSDataset) -> dict:
    """Feature computation time per sample."""
    t0 = time.perf_counter()
    n = min(200, len(ds))
    for j in range(n):
        ds.features(j)
    per = (time.perf_counter() - t0) / n
    return {"features_ms_per_sample": per * 1e3,
            "epoch_50k_single_process_min": per * 50000 / 60,
            "epoch_50k_8_workers_min_est": per * 50000 / 60 / 8}


def main() -> None:
    """Write the Phase 4 figures and summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic/v1/pilot.h5")
    ap.add_argument("--out", default="reports/phase4")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    ds = TPSDataset(PROJECT_ROOT / args.data, train=True, base_seed=1001)
    summary = {"data": args.data, "n": len(ds)}
    summary["realisation_samples"] = realisations(ds, out)
    _, summary["normalised"] = normalised(ds, out)
    summary["m0_noisy_gate"] = m0_gate(out)
    sensitivity(out)
    summary["timing"] = timing(ds)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
