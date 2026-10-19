"""Phase 1 validation: D(tau) against reference values and the M0 gate on noise-free analytic curves.

Writes figures and a CSV to reports/phase1/ and prints a summary.
Usage: python scripts/validate_m0.py [--config configs/analysis.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tpslib.config import load_sensor, load_yaml
from tpslib.dtau import SIGMA_MIN, dtau, dtau_exact, small_sigma_slope
from tpslib.fit_tps import PowerHistory, fit_tps
from tpslib.runtime import PROJECT_ROOT
from tpslib.synth_analytic import make_case

# Reference categorical palette (slots 1-2) + recessive ink
C1, C2 = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REFERENCE = {0.1: 0.20553, 0.3: 0.35460, 0.5: 0.46487, 0.75: 0.55840, 1.0: 0.61900, 1.5: 0.68912}


def style(ax):
    """Report axis style."""
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main() -> None:
    """D(tau) checks and the M0 gate; figures and CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/analysis.yaml")
    ap.add_argument("--out", default="reports/phase1")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    s = load_sensor()
    a, m = s.outer_radius, s.n_rings
    summary: dict = {"config": cfg}

    # --- D(tau) vs reference --------------------------------------------------
    rows = []
    for tau, ref in REFERENCE.items():
        ex = float(dtau_exact(tau, m)[0])
        tb = float(dtau(np.array([tau]), m)[0])
        rows.append({"tau": tau, "reference": ref, "exact": ex, "table": tb,
                     "err_exact_pct": (ex / ref - 1) * 100, "err_table_vs_exact": tb - ex})
    dref = pd.DataFrame(rows)
    dref.to_csv(out / "dtau_reference.csv", index=False)
    tau_dense = np.geomspace(2 * SIGMA_MIN, 10, 300)
    table_err = float(np.max(np.abs(dtau(tau_dense, m) - dtau_exact(tau_dense, m))))
    summary["dtau"] = {"max_ref_err_pct": float(dref.err_exact_pct.abs().max()), "max_table_abs_err": table_err}

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    tt = np.geomspace(1e-3, 3, 400)
    axs[0].plot(tt, dtau(tt, m), color=C1, lw=2, label="D(τ), m = 20")
    axs[0].plot(list(REFERENCE), list(REFERENCE.values()), "o", ms=8, mfc="white", mec=C2, mew=2,
                label="reference values")
    axs[0].set_xscale("log")
    axs[0].set_xlabel("τ")
    axs[0].set_ylabel("D(τ)")
    axs[0].legend(frameon=False)
    axs[0].set_title("Gustafsson D(τ)", color=INK, loc="left")
    style(axs[0])
    sig = np.geomspace(1e-7, 3, 400)
    axs[1].plot(sig, np.ones_like(sig) * small_sigma_slope(m), color=INK2, lw=1, ls="--",
                label="1/(2√π(m+1))")
    from tpslib.dtau import _ring_sum
    axs[1].plot(sig, _ring_sum(sig, m), color=C1, lw=2, label="σ·f(σ)")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("σ")
    axs[1].set_ylabel("ln σ cinsinden integrand")
    axs[1].set_title("σ→0'da sabite gider: integral ln σ gibi ıraksar", color=INK, loc="left", fontsize=10)
    axs[1].legend(frameon=False)
    style(axs[1])
    fig.tight_layout()
    fig.savefig(out / "dtau.png", dpi=150)
    plt.close(fig)

    # --- M0 gate --------------------------------------------------------------
    k_values = np.geomspace(0.05, 30.0, 20)
    rho_cp = np.linspace(0.8e6, 4.5e6, 20)[np.argsort(np.random.default_rng(0).random(20))]
    rows = []
    for k, rc in zip(k_values, rho_cp):
        alpha = k / rc
        for mode in ("sabit güç", "sabit akım"):
            cc = mode == "sabit akım"
            c = make_case(k, alpha, constant_current=cc)
            t0 = time.perf_counter()
            r = fit_tps(c.t, c.dT, c.P_t if cc else c.P, a, m)
            dt = time.perf_counter() - t0
            row = {"k": k, "alpha": alpha, "rho_cp": rc, "mode": mode, "P": c.P, "sps": c.sps,
                   "n_samples": c.t.size, "t_end": c.t[-1], "dT_max": c.dT.max(),
                   "k_fit": r.k, "alpha_fit": r.alpha, "err_k_pct": (r.k / k - 1) * 100,
                   "err_alpha_pct": (r.alpha / alpha - 1) * 100, "converged": r.converged,
                   "fit_time_s": dt}
            if cc:  # what happens if constant current is ignored (mean power, constant-power model)
                rn = fit_tps(c.t, c.dT, float(np.mean(c.P_t)), a, m)
                row["err_k_naive_pct"] = (rn.k / k - 1) * 100
                row["err_alpha_naive_pct"] = (rn.alpha / alpha - 1) * 100
            rows.append(row)
    g = pd.DataFrame(rows)
    g.to_csv(out / "m0_gate.csv", index=False)
    gate_ok = bool((g.err_k_pct.abs() < 0.5).all() and (g.err_alpha_pct.abs() < 0.5).all() and g.converged.all())
    summary["m0_gate"] = {
        "passed": gate_ok,
        "max_abs_err_k_pct": float(g.err_k_pct.abs().max()),
        "max_abs_err_alpha_pct": float(g.err_alpha_pct.abs().max()),
        "naive_max_abs_err_k_pct": float(g.err_k_naive_pct.abs().max()),
        "naive_max_abs_err_alpha_pct": float(g.err_alpha_naive_pct.abs().max()),
        "max_fit_time_s": float(g.fit_time_s.max()),
    }

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, col, name in [(axs[0], "err_k_pct", "k"), (axs[1], "err_alpha_pct", "α")]:
        ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0)
        for mode, c, mk in [("sabit güç", C1, "o"), ("sabit akım", C2, "s")]:
            d = g[g["mode"] == mode]
            ax.plot(d.k, d[col], mk, color=c, ms=8, mec="white", mew=1, label=mode, zorder=3)
        ax.set_xscale("log")
        ax.set_ylim(-0.6, 0.6)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xlabel("k [W/m·K]")
        ax.set_ylabel(f"{name} bağıl hata [%]")
        ax.set_title(f"M0 {name} hatası (gri bant: ±%0,5 kapı)", color=INK, loc="left", fontsize=10)
        style(ax)
    axs[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "m0_gate_errors.png", dpi=150)
    plt.close(fig)

    d = g[g["mode"] == "sabit akım"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, (good, naive, name) in zip(axs, [("err_k_pct", "err_k_naive_pct", "k"),
                                              ("err_alpha_pct", "err_alpha_naive_pct", "α")]):
        ax.plot(d.k, d[naive], "s", color=C2, ms=8, mec="white", mew=1, label="P sabit varsayılırsa", zorder=3)
        ax.plot(d.k, d[good], "o", color=C1, ms=8, mec="white", mew=1, label="P(t) hesaba katılırsa (M0)", zorder=3)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("k [W/m·K]")
        ax.set_ylabel(f"{name} bağıl hata [%]")
        ax.set_title(f"Sabit akımda güç değişiminin {name}'ya etkisi", color=INK, loc="left", fontsize=10)
        style(ax)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "m0_constant_current_effect.png", dpi=150)
    plt.close(fig)

    # --- Example fit (PMMA-like, constant current) -----------------------------
    k, alpha = 0.19, 0.13e-6
    c = make_case(k, alpha, constant_current=True)
    r = fit_tps(c.t, c.dT, c.P_t, a, m)
    hist = PowerHistory(c.t, c.P_t, int(cfg["m0"]["n_power_steps"]))
    Dw = hist.d_weighted(c.t, r.alpha, a, m)
    model = r.A * Dw + r.B * hist.p_norm(c.t)
    win = (c.t >= r.t_window[0]) & (c.t <= r.t_window[1])
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot(Dw, c.dT, color=INK2, lw=1, label="tüm kayıt")
    axs[0].plot(Dw[win], c.dT[win], color=C1, lw=3, label="uydurma penceresi 0,3<τ²<1")
    axs[0].set_xlabel("D̃(τ) (güç geçmişiyle ağırlıklı)")
    axs[0].set_ylabel("ΔT [K]")
    axs[0].set_title(f"PMMA benzeri: k={k}, α=0,13 mm²/s → hata %{(r.k/k-1)*100:+.4f}",
                     color=INK, loc="left", fontsize=10)
    axs[0].legend(frameon=False)
    style(axs[0])
    axs[1].plot(c.t[win], (c.dT[win] - model[win]) * 1e6, color=C1, lw=1.5)
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("artık [µK]")
    axs[1].set_title("Penceredeki artık (gürültüsüz)", color=INK, loc="left", fontsize=10)
    style(axs[1])
    fig.tight_layout()
    fig.savefig(out / "m0_example_fit.png", dpi=150)
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
