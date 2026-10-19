"""Phase 2: simulator validation gates and figures.

Gates
  1. Analytic limit : no film, perfect contact, constant power, large sample  vs  finite-band analytic
                      solution (and D(tau)); pointwise within 0.5 % for 0.1 <= tau <= 1.
  2. Convergence    : mesh and time step halved (refine=2); change < 0.1 %.
  3. Energy         : energy in = stored (all cells) + convective loss, within 0.5 %.
  4. Physics        : lower h_c -> lower apparent k; thin sample -> late deviation; film -> early deviation.
  5. Speed          : single core time per simulation, estimate for 60 000 simulations.

Usage: python scripts/validate_sim.py [--out reports/phase2] [--quick]
"""

from __future__ import annotations

import os

# Single-core timing: must be set before numpy/scipy are imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np
import pandas as pd

from tpslib.config import load_sensor
from tpslib.dtau import PI15, dtau
from tpslib.eval.report import BAND, INK2, MARKERS, SERIES, plt, save, style
from tpslib.fit_tps import fit_tps
from tpslib.runtime import PROJECT_ROOT
from tpslib.sim.analytic import band_reference
from tpslib.sim.sensor_model import SimParams, load_numerics, simulate
from tpslib.synth_analytic import setup_power_and_rate

SENSOR = load_sensor()
A, M, B = SENSOR.outer_radius, SENSOR.n_rings, SENSOR.trace_width
NU = load_numerics()
# Reference materials of the project plan (k [W/m/K], rho*cp [J/m^3/K]) + range ends
MATERIALS = {
    "PMMA": (0.19, 1.46e6),
    "Borosilicate": (1.2, 2.0e6),
    "Stainless steel 304": (16.0, 3.7e6),
}
EXTRA = {"Foam-like (0.05)": (0.05, 1.0e6), "Upper limit (30)": (30.0, 3.5e6)}


def nominal(k, rc, **kw) -> SimParams:
    """Nominal parameters for a material with the auto-setup power and duration."""
    P, t_end, sps = setup_power_and_rate(k, k / rc, A, M)
    base = dict(k=k, rho_cp=rc, P0=P, t_end=t_end, h_c_bottom=5000.0, h_c_top=5000.0)
    base.update(kw)
    return SimParams(**base)


def tau_of(p: SimParams, t):
    """Dimensionless time sqrt(alpha t) / a."""
    return np.sqrt(p.alpha * t) / A


def m0(r, p):
    """Classic TPS fit of a simulation result."""
    return fit_tps(r.t, r.dT, r.P if p.constant_current else p.P0, A, M)


def gate_analytic(out, summary):
    """Gate 1: simulator vs the finite-band analytic solution."""
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, (k, rc)) in enumerate(MATERIALS.items()):
        g = NU["analytic_limit"]
        p = nominal(k, rc, layers=False, h_c_bottom=np.inf, h_c_top=np.inf, constant_current=False,
                    h_convection=0.0, sample_radius=g["sample_radius"], thickness_bottom=g["sample_thickness"],
                    thickness_top=g["sample_thickness"])
        r = simulate(p)
        an = band_reference(r.t, k, p.alpha, p.P0, A, M, B)
        tau = tau_of(p, r.t)
        e = r.dT / an - 1
        f = m0(r, p)
        win = (tau >= 0.1) & (tau <= 1.0)
        rows.append({"material": name, "k": k, "max_err_tau_0.1_1_pct": np.abs(e[win]).max() * 100,
                     "max_err_tau_0.05_1_pct": np.abs(e[tau >= 0.05]).max() * 100,
                     "max_err_all_pct": np.abs(e).max() * 100,
                     "M0_err_k_pct": (f.k / k - 1) * 100, "M0_err_alpha_pct": (f.alpha / p.alpha - 1) * 100})
        c, mk = SERIES[i], MARKERS[i]
        axs[1].plot(tau, e * 100, color=c, lw=2, label=name)
        if i == 0:
            axs[0].plot(tau, an, color=INK2, lw=4, alpha=0.35, label="analytic (finite trace width)")
            axs[0].plot(tau, r.dT, color=c, lw=1.5, label="simulator")
            Dshift = p.P0 / (PI15 * A * k) * dtau(tau)
            off = np.mean((an - Dshift)[win])
            axs[0].plot(tau, Dshift + off, ":", color=SERIES[1], lw=2, label="D(τ) + const")
    axs[0].set_xscale("log"); axs[0].set_xlabel("τ"); axs[0].set_ylabel("ΔT [K]")
    axs[0].legend(frameon=False)
    style(axs[0], "Analytic limit, PMMA: no layers, perfect contact, constant power")
    axs[1].axhspan(-0.5, 0.5, color=BAND, zorder=0)
    axs[1].axvspan(0.1, 1.0, color="#dbe8f8", alpha=0.5, zorder=0)
    axs[1].set_xscale("log"); axs[1].set_ylim(-2, 2)
    axs[1].set_xlabel("τ"); axs[1].set_ylabel("relative difference [%]")
    axs[1].legend(frameon=False)
    style(axs[1], "Simulator / analytic − 1 (grey: ±0.5%; blue: gate window)")
    save(fig, out / "gate1_analytic.png")
    d = pd.DataFrame(rows)
    d.to_csv(out / "gate1_analytic.csv", index=False)
    summary["gate1_analytic"] = {"passed": bool((d["max_err_tau_0.1_1_pct"] < 0.5).all()), "rows": rows}


def gate_convergence(out, summary):
    """Gate 2: change when mesh and time step are halved."""
    rows = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (name, (k, rc)) in enumerate(MATERIALS.items()):
        p = nominal(k, rc)
        r1 = simulate(p)
        r2 = simulate(p, refine=2)
        nu_t = json.loads(json.dumps(NU))
        nu_t["time"]["dt0"] /= 2
        nu_t["time"]["steps_per_level"] *= 2
        rt = simulate(p, numerics=nu_t)
        tau = tau_of(p, r1.t)
        win = (tau >= 0.1) & (tau <= 1.0)
        c = r2.dT / r1.dT - 1
        ct = rt.dT / r1.dT - 1
        dd = (r2.dT - r1.dT)[win]
        f1, f2 = m0(r1, p), m0(r2, p)
        rows.append({
            "material": name,
            "max_change_tau_0.1_1_pct": np.abs(c[win]).max() * 100,
            "max_change_all_pct": np.abs(c).max() * 100,
            "time_only_change_tau_0.1_1_pct": np.abs(ct[win]).max() * 100,
            "offset_part_K": float(dd.mean()),
            "shape_change_after_offset_pct": float(np.abs(dd - dd.mean()).max() / r1.dT[win].max() * 100),
            "M0_k_change_pct": (f2.k / f1.k - 1) * 100,
            "M0_alpha_change_pct": (f2.alpha / f1.alpha - 1) * 100,
            "time_base_s": r1.info["time_total_s"], "time_refined_s": r2.info["time_total_s"],
            "n_unknowns_base": r1.info["n_unknowns"], "n_unknowns_refined": r2.info["n_unknowns"],
        })
        ax.plot(tau, c * 100, color=SERIES[i], lw=2, label=f"{name}: mesh + time")
        ax.plot(tau, ct * 100, "--", color=SERIES[i], lw=1.2, label=f"{name}: time only")
    ax.axhspan(-0.1, 0.1, color=BAND, zorder=0)
    ax.set_xscale("log"); ax.set_xlabel("τ"); ax.set_ylabel("change [%]")
    ax.legend(frameon=False, fontsize=8)
    style(ax, "ΔT change when mesh and time step are halved (grey: ±0.1%)")
    save(fig, out / "gate2_convergence.png")
    d = pd.DataFrame(rows)
    d.to_csv(out / "gate2_convergence.csv", index=False)
    summary["gate2_convergence"] = {"passed": bool((d["max_change_tau_0.1_1_pct"] < 0.1).all()), "rows": rows}


def gate_energy(out, summary):
    """Gate 3: energy in = stored + lost."""
    rows = []
    cases = [(n, k, rc, {}) for n, (k, rc) in MATERIALS.items()]
    cases.append(("PMMA, low h_c, constant power", 0.19, 1.46e6, dict(h_c_bottom=500.0, h_c_top=800.0, constant_current=False)))
    cases.append(("PMMA, thin sample 8 mm", 0.19, 1.46e6, dict(thickness_bottom=8e-3, thickness_top=8e-3)))
    for name, k, rc, kw in cases:
        p = nominal(k, rc, **kw)
        r = simulate(p)
        e = r.energy
        # Independent check: trapezoid integral of the recorded power over the raw step times
        cfb = SENSOR.alpha_R if p.constant_current else 0.0
        Ps = p.P0 * (1 + cfb * r.dT_steps)
        ts = np.concatenate([[0.0], r.t_steps])
        Pt = np.concatenate([[p.P0], Ps])
        E_trap = float(np.sum(0.5 * (Pt[1:] + Pt[:-1]) * np.diff(ts)))
        rows.append({"case": name, "E_in_J": e["E_in"], "E_stored_J": e["E_stored"], "E_loss_J": e["E_loss"],
                     "rel_imbalance": e["rel_imbalance"],
                     "rel_imbalance_trapezoid": (E_trap - e["E_stored"] - e["E_loss"]) / E_trap,
                     "frac_samples": e["frac_stored_samples"], "frac_film": e["frac_stored_film"],
                     "frac_lost": e["frac_lost"]})
    d = pd.DataFrame(rows)
    d.to_csv(out / "gate3_energy.csv", index=False)
    summary["gate3_energy"] = {"passed": bool((d.rel_imbalance.abs() < 5e-3).all()
                                              and (d.rel_imbalance_trapezoid.abs() < 5e-3).all()), "rows": rows}


def physics_hc(out, summary):
    """Apparent k and alpha (M0) vs contact conductance."""
    hcs = np.geomspace(500, 50000, 9)
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for i, (name, (k, rc)) in enumerate(MATERIALS.items()):
        ek, ea = [], []
        for hc in hcs:
            p = nominal(k, rc, h_c_bottom=hc, h_c_top=hc)
            r = simulate(p)
            f = m0(r, p)
            ek.append((f.k / k - 1) * 100); ea.append((f.alpha / p.alpha - 1) * 100)
            rows.append({"material": name, "h_c": hc, "M0_k_err_pct": ek[-1], "M0_alpha_err_pct": ea[-1],
                         "dT_end": r.dT[-1], "M0_B": f.B})
        axs[0].plot(hcs, ek, MARKERS[i] + "-", color=SERIES[i], ms=7, mec="white", label=name)
        axs[1].plot(hcs, ea, MARKERS[i] + "-", color=SERIES[i], ms=7, mec="white", label=name)
    for ax, lab in zip(axs, ["k", "α"]):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log"); ax.set_xlabel("h_c (both faces) [W/m²·K]"); ax.set_ylabel(f"M0 apparent {lab} error [%]")
        ax.legend(frameon=False)
        style(ax, f"Apparent {lab} as the contact conductance drops")
    save(fig, out / "physics_hc.png")
    d = pd.DataFrame(rows)
    d.to_csv(out / "physics_hc.csv", index=False)
    mono = all(np.all(np.diff(d[d.material == n].sort_values("h_c").M0_k_err_pct) > -1e-3) for n in MATERIALS)
    summary["physics_hc"] = {"apparent_k_increases_with_hc": bool(mono)}


def physics_thickness(out, summary):
    """Late-time deviation of thin samples."""
    th = [8e-3, 12e-3, 16e-3, 20e-3, 25e-3]
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axs, ["PMMA", "Borosilicate"]):
        k, rc = MATERIALS[name]
        ref = simulate(nominal(k, rc, thickness_bottom=0.08, thickness_top=0.08, sample_radius=0.08))
        for i, H in enumerate(th):
            p = nominal(k, rc, thickness_bottom=H, thickness_top=H)
            r = simulate(p)
            tau = tau_of(p, r.t)
            dev = (r.dT / ref.dT - 1) * 100
            f = m0(r, p)
            # first tau where the deviation exceeds 0.1 %
            over = np.where(np.abs(dev) > 0.1)[0]
            onset = float(tau[over[0]]) if over.size else np.nan
            rows.append({"material": name, "thickness_mm": H * 1e3, "onset_tau_0.1pct": onset,
                         "dev_at_tau1_pct": dev[-1], "M0_k_err_pct": (f.k / k - 1) * 100,
                         "M0_alpha_err_pct": (f.alpha / p.alpha - 1) * 100,
                         "depth_over_thickness_tau1": 2 * A / H})
            ax.plot(tau, dev, color=SERIES[i], lw=2, label=f"{H*1e3:.0f} mm")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvspan(np.sqrt(0.3), 1.0, color=BAND, zorder=0)
        ax.set_xscale("log"); ax.set_xlabel("τ"); ax.set_ylabel("ΔT deviation vs 80 mm [%]")
        ax.legend(frameon=False, title="sample thickness", fontsize=8)
        style(ax, f"{name}: late-time deviation for thin samples (grey: M0 window)")
    save(fig, out / "physics_thickness.png")
    d = pd.DataFrame(rows)
    d.to_csv(out / "physics_thickness.csv", index=False)
    summary["physics_thickness"] = {"rows": rows}


def physics_layers(out, summary):
    """Early-time effect of the film and the contacts."""
    k, rc = MATERIALS["Borosilicate"]
    cases = [
        ("no film, perfect contact", dict(layers=False, h_c_bottom=np.inf, h_c_top=np.inf)),
        ("film, perfect contact", dict(h_c_bottom=np.inf, h_c_top=np.inf)),
        ("film, h_c = 5000", dict()),
        ("film, h_c = 1000", dict(h_c_bottom=1000.0, h_c_top=1000.0)),
    ]
    res = {}
    for name, kw in cases:
        p = nominal(k, rc, constant_current=False, **kw)
        res[name] = (p, simulate(p))
    p0, r0 = res[cases[0][0]]
    tau = tau_of(p0, r0.t)
    lt = np.log(r0.t)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for i, (name, _) in enumerate(cases):
        p, r = res[name]
        axs[0].plot(tau, r.dT, color=SERIES[i], lw=2, label=name)
        slope = np.gradient(r.dT, lt) / np.gradient(r0.dT, lt) - 1
        if i > 0:
            axs[1].plot(tau, slope * 100, color=SERIES[i], lw=2, label=name)
            over = np.where(np.abs(slope) > 0.01)[0]
            rows.append({"case": name, "offset_at_tau1_K": float(r.dT[-1] - r0.dT[-1]),
                         "slope_dev_1pct_until_tau": float(tau[over[-1]]) if over.size else np.nan,
                         "slope_dev_at_tau_0.55_pct": float(np.interp(np.sqrt(0.3), tau, slope) * 100)})
    axs[0].set_xscale("log"); axs[0].set_xlabel("τ"); axs[0].set_ylabel("ΔT [K]")
    axs[0].legend(frameon=False, fontsize=8)
    style(axs[0], "Borosilicate, constant power: effect of film and contact")
    axs[1].axhline(0, color=INK2, lw=0.8)
    axs[1].axvspan(np.sqrt(0.3), 1.0, color=BAND, zorder=0)
    axs[1].set_xscale("log"); axs[1].set_yscale("symlog", linthresh=1.0)
    axs[1].set_xlabel("τ"); axs[1].set_ylabel("dΔT/d ln t deviation [%]")
    axs[1].legend(frameon=False, fontsize=8)
    style(axs[1], "Slope deviation at early times (grey: M0 window)")
    save(fig, out / "physics_layers.png")
    pd.DataFrame(rows).to_csv(out / "physics_layers.csv", index=False)
    summary["physics_layers"] = {"rows": rows}


def speed(out, summary):
    """Single-core run time per simulation."""
    rows = []
    for name, (k, rc) in {**MATERIALS, **EXTRA}.items():
        p = nominal(k, rc)
        P, t_end, sps = setup_power_and_rate(k, k / rc, A, M)
        p.t_out_min = 1.0 / sps
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = simulate(p)
            ts.append(time.perf_counter() - t0)
        rows.append({"material": name, "k": k, "t_end_s": t_end, "sps": sps, "time_s": float(np.median(ts)),
                     "n_unknowns": r.info["n_unknowns"], "n_steps": r.info["n_steps"],
                     "n_factorizations": r.info["n_factorizations"]})
    d = pd.DataFrame(rows)
    d.to_csv(out / "speed.csv", index=False)
    mean_t = float(d.time_s.mean())
    summary["speed"] = {"mean_s": mean_t, "max_s": float(d.time_s.max()), "passed_1s": bool(d.time_s.max() < 1.0),
                        "est_60k_single_core_h": mean_t * 60000 / 3600,
                        "est_60k_14_processes_h": mean_t * 60000 / 3600 / 14, "rows": rows}


def main() -> None:
    """Run the selected validation steps and write summary.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/phase2")
    ap.add_argument("--only", nargs="*", help="run only these steps, e.g. physics_layers")
    args = ap.parse_args()
    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary: dict = json.loads(summary_path.read_text(encoding="utf-8")) if args.only and summary_path.exists() else {}
    steps = (gate_analytic, gate_energy, speed, physics_layers, physics_thickness, physics_hc, gate_convergence)
    for fn in steps:
        if args.only and fn.__name__ not in args.only:
            continue
        t0 = time.perf_counter()
        fn(out, summary)
        print(f"{fn.__name__}: {time.perf_counter() - t0:.0f} s", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                          encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in summary.items()},
                     indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
