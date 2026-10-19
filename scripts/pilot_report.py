"""Phase 3 pilot report: parameter / auto-setup histograms, dT_max distribution, timing, example curves.

Usage: python scripts/pilot_report.py [--split pilot] [--out reports/phase3] [--workers 14]
"""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np

from tpslib.config import CONFIG_DIR, load_sensor, load_yaml
from tpslib.dtau import PI15
from tpslib.eval.report import BAND, INK2, SERIES, plt, save, style
from tpslib.data.generate import dataset_dir
from tpslib.runtime import PROJECT_ROOT


def hist(ax, x, title, xlabel, log=False, bins=40, color=SERIES[0]):
    """Histogram helper (linear or log bins)."""
    x = np.asarray(x)
    b = np.geomspace(x.min(), x.max(), bins) if log else bins
    ax.hist(x, bins=b, color=color, edgecolor="white", linewidth=0.6)
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("örnek sayısı")
    style(ax, title)


def main() -> None:
    """Write the pilot figures and summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_v1.yaml")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--out", default="reports/phase3")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    a = load_sensor().outer_radius
    with h5py.File(dataset_dir(cfg) / f"{args.split}.h5", "r") as f:
        p = {k: f["params"][k][()] for k in f["params"]}
        t, dT, P = f["t"][()], f["dT"][()].astype(float), f["P"][()].astype(float)
        diag = {k: f["diag"][k][()] for k in f["diag"]}
        stats = json.loads(f.attrs["stats"])
    ok = diag["ok"] == 1
    dTmax = dT[:, -1]

    # --- parameters ---------------------------------------------------------------------
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    ax = axs.ravel()
    hist(ax[0], p["k"], "k (log-düzgün)", "k [W/m·K]", log=True)
    hist(ax[1], p["rho_cp"] / 1e6, "ρc_p (düzgün)", "ρc_p [MJ/m³·K]")
    hist(ax[2], p["alpha"] * 1e6, "α = k/ρc_p", "α [mm²/s]", log=True)
    hist(ax[3], p["h_c_bottom"], "h_c alt (log-düzgün)", "h_c [W/m²·K]", log=True)
    hist(ax[4], p["h_c_top"], "h_c üst (log-düzgün)", "h_c [W/m²·K]", log=True)
    hist(ax[5], p["G_c"], "G_c: toplam arayüz iletkenliği (model hedefi)", "G_c [W/m²·K]", log=True, color=SERIES[1])
    hist(ax[6], p["sample_thickness"] * 1e3, "numune kalınlığı", "[mm]")
    hist(ax[7], p["sample_radius"] * 1e3, "numune yarıçapı", "[mm]")
    hist(ax[8], p["k_polyimide"], "PI k", "[W/m·K]")
    hist(ax[9], p["k_adhesive"], "yapıştırıcı k", "[W/m·K]")
    hist(ax[10], p["scale_polyimide_base"], "PI taban kalınlık çarpanı", "× nominal")
    hist(ax[11], p["alpha_R_calib_rel"] * 100, "α_R kalibrasyon hatası", "[%]")
    save(fig, out / "pilot_params.png")

    # --- auto-setup ---------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    ax = axs.ravel()
    hist(ax[0], p["P0"] * 1e3, "güç P₀ (otomatik ayar)", "P₀ [mW]", log=True, color=SERIES[2])
    hist(ax[1], p["t_end"], "ölçüm süresi t_son", "t_son [s]", log=True, color=SERIES[2])
    rates, counts = np.unique(p["sps"], return_counts=True)
    ax[2].bar([str(int(r)) for r in rates], counts, color=SERIES[2], edgecolor="white")
    ax[2].set_xlabel("veri hızı [SPS]"); ax[2].set_ylabel("örnek sayısı")
    style(ax[2], "veri hızı")
    hist(ax[3], p["tau_end"], "gerçek τ_son (yoklama hatası ±%30)", "τ_son", color=SERIES[2])
    rng_counts = [int((p["current_range"] == 0).sum()), int((p["current_range"] == 1).sum())]
    ax[4].bar(["L (45–188 mA)", "H (180–500 mA)"], rng_counts, color=SERIES[2], edgecolor="white")
    ax[4].set_ylabel("örnek sayısı")
    style(ax[4], "akım aralığı")
    ax[5].scatter(p["k"], p["P0"] * 1e3, s=6, color=SERIES[2], alpha=0.6)
    ax[5].set_xscale("log"); ax[5].set_yscale("log")
    ax[5].set_xlabel("k [W/m·K]"); ax[5].set_ylabel("P₀ [mW]")
    style(ax[5], "güç k ile ölçeklenir; 8 mW / 1 W'ta kırpılır")
    save(fig, out / "pilot_autosetup.png")

    # --- dT_max -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    hist(axs[0], dTmax[ok], "ΔT_max dağılımı", "ΔT_max [K]", bins=50, color=SERIES[1])
    axs[0].axvspan(0.5, 8.0, color=BAND, zorder=0)
    sc = axs[1].scatter(p["k"], dTmax, c=np.log10(p["G_c"]), s=6, cmap="viridis")
    axs[1].set_xscale("log"); axs[1].set_xlabel("k [W/m·K]"); axs[1].set_ylabel("ΔT_max [K]")
    cb = fig.colorbar(sc, ax=axs[1]); cb.set_label("log₁₀ G_c")
    style(axs[1], "ΔT_max – k (renk: arayüz iletkenliği)")
    axs[2].scatter(p["G_c"], dTmax - 2.5, s=6, color=SERIES[1], alpha=0.6)
    axs[2].set_xscale("log"); axs[2].set_xlabel("G_c [W/m²·K]"); axs[2].set_ylabel("ΔT_max − 2,5 K")
    style(axs[2], "Hedeften sapma: film + temas ofseti")
    save(fig, out / "pilot_dTmax.png")

    # --- timing -------------------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    hist(axs[0], diag["sim_time_s"], "simülasyon süresi (süreç başına)", "[s]", bins=40, color=SERIES[3])
    axs[1].scatter(p["t_end"], diag["sim_time_s"], s=6, color=SERIES[3], alpha=0.6)
    axs[1].set_xscale("log"); axs[1].set_xlabel("t_son [s]"); axs[1].set_ylabel("süre [s]")
    style(axs[1], "süre – ölçüm uzunluğu")
    save(fig, out / "pilot_timing.png")

    # --- example curves -----------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    idx = np.argsort(p["k"])[np.linspace(0, len(p["k"]) - 1, 12).astype(int)]
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(idx):
        tau = np.sqrt(p["alpha"][i] * t[i]) / a
        col = cmap(j / (len(idx) - 1))
        axs[0].plot(t[i], dT[i], color=col, lw=1.4)
        theta = dT[i] * PI15 * a / P[i]
        axs[1].plot(tau, theta * p["k"][i], color=col, lw=1.4, label=f"k={p['k'][i]:.2g}")
    axs[0].set_xscale("log"); axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("ΔT [K]")
    style(axs[0], "12 örnek temiz eğri (renk: k)")
    axs[1].set_xscale("log"); axs[1].set_xlabel("τ (gerçek α ile)"); axs[1].set_ylabel("k·θ  (ideal durumda D(τ) + sabit)")
    axs[1].legend(frameon=False, fontsize=7, ncol=2)
    style(axs[1], "Normalize eğriler: farklar film + temas ofsetinden")
    save(fig, out / "pilot_curves.png")

    # --- summary ------------------------------------------------------------------------
    full = {name: int(sc["n"]) for name, sc in cfg["splits"].items()}
    n_full = sum(full.values())
    per_sim = float(np.mean(diag["sim_time_s"]))
    wall_per_sample = stats["wall_time_s_this_run"] / stats["n"]
    summary = {
        "n": int(len(ok)), "n_ok": int(ok.sum()), "n_failed": int((~ok).sum()),
        "wall_time_s": stats["wall_time_s_this_run"], "workers": stats["workers"],
        "sim_time_mean_s": per_sim, "sim_time_p95_s": float(np.percentile(diag["sim_time_s"], 95)),
        "effective_speedup": per_sim / wall_per_sample,
        "full_run_samples": full, "full_run_estimate_h": wall_per_sample * n_full / 3600,
        "dTmax": {"median": float(np.median(dTmax)), "p5": float(np.percentile(dTmax, 5)),
                  "p95": float(np.percentile(dTmax, 95)),
                  "frac_below_0.5K": float(np.mean(dTmax < 0.5)), "frac_above_8K": float(np.mean(dTmax > 8.0))},
        "power_clipped_low": float(np.mean(p["P0"] <= 0.008 * 1.0001)),
        "power_clipped_high": float(np.mean(p["P0"] >= 1.0 * 0.9999)),
        "t_end_limited_by_depth": float(np.mean(p["t_end_limited_by_depth"])),
        "sps_counts": {str(int(r)): int(c) for r, c in zip(rates, counts)},
        "current_range_counts": {"L": rng_counts[0], "H": rng_counts[1]},
        "energy_imbalance_max": float(np.nanmax(np.abs(diag["energy_rel_imbalance"]))),
        "frac_lost_max": float(np.nanmax(diag["frac_lost"])),
    }
    (out / "pilot_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
