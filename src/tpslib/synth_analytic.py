"""Noise-free analytic TPS records for code verification (Phase 1 gate, T5 in the project plan).

Power and duration follow a simplified version of the device auto-setup rule with the TRUE k and
alpha (no probe error): power gives the target rise at tau = 1, duration reaches tau = tau_end,
sample rate is the fastest allowed rate within the per-record sample budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpslib.config import CONFIG_DIR, load_sensor, load_yaml
from tpslib.dtau import PI15, dtau, ideal_rise_constant_current, ideal_rise_constant_power

_DEV = load_yaml(CONFIG_DIR / "device.yaml")


@dataclass
class AnalyticCase:
    """Noise-free analytic record sampled at the device rate."""
    k: float
    alpha: float
    P: float          # initial power [W]
    sps: int
    t: np.ndarray     # [s]
    dT: np.ndarray    # [K]
    P_t: np.ndarray   # power at each sample [W]


def setup_power_and_rate(k, alpha, a, m, tau_end=1.0):
    """Auto-setup with the true k and alpha: power, duration and sample rate."""
    P = float(_DEV["target_temperature_rise"]) * PI15 * a * k / float(dtau(np.array(1.0), m))
    P = float(np.clip(P, _DEV["power"]["min"], _DEV["power"]["max"]))
    t_end = tau_end**2 * a * a / alpha
    cap = int(_DEV["sample_rate"]["max_samples_per_record"])
    allowed = sorted(_DEV["sample_rate"]["allowed"])
    sps = max([r for r in allowed if r * t_end <= cap], default=allowed[0])
    return P, t_end, sps


def make_case(k, alpha, constant_current=False, tau_end=1.0, sensor=None) -> AnalyticCase:
    """Analytic record (constant power or constant current) for the M0 code-verification gate."""
    s = sensor or load_sensor()
    a, m = s.outer_radius, s.n_rings
    P, t_end, sps = setup_power_and_rate(k, alpha, a, m, tau_end)
    t = np.arange(1, int(np.floor(t_end * sps)) + 1) / sps
    if not constant_current:
        return AnalyticCase(k, alpha, P, sps, t, ideal_rise_constant_power(t, k, alpha, a, P, m), np.full_like(t, P))
    # Constant current: solve on a dense log grid, then interpolate to the sample times.
    I = np.sqrt(P / s.R0)
    tg = np.geomspace(t[0] / 4, t[-1], 1500)
    dTg, Pg = ideal_rise_constant_current(tg, k, alpha, a, I, s.R0, s.alpha_R, m)
    lt = np.log(t)
    return AnalyticCase(k, alpha, P, sps, t, np.interp(lt, np.log(tg), dTg), np.interp(lt, np.log(tg), Pg))


def clean_record_case(k: float, rho_cp: float = 2.0e6, n_grid: int = 1500, sensor=None, **extra):
    """Noise-free constant-current analytic curve plus a prior-style parameter row.

    Returns (t, dT, params) in the format of a simulated dataset sample, so the measurement chain
    (`measurement.acquisition`) and the estimators can be exercised against an exact reference.
    Keyword arguments in `extra` override parameter-row entries (e.g. calibration errors, faults).
    """
    s = sensor or load_sensor()
    a, m = s.outer_radius, s.n_rings
    alpha = k / rho_cp
    P, t_end, sps = setup_power_and_rate(k, alpha, a, m)
    I = np.sqrt(P / s.R0)
    t = np.geomspace(0.25 / sps, t_end, n_grid)
    dT, _ = ideal_rise_constant_current(t, k, alpha, a, I, s.R0, s.alpha_R, m)
    prm = {"k": k, "alpha": alpha, "P0": P, "sps": sps, "t_end": t_end,
           "current_range": 0 if I <= _DEV["current_ranges"]["L"]["I_max"] else 1,
           "alpha_R_calib_rel": 0.0, "radius_calib_rel": 0.0, "thickness_entered": 0.02,
           "sample_radius": 0.025, "drift_jump_K": 0.0, "drift_jump_t_frac": 0.0, "G_c": 5000.0}
    prm.update(extra)
    return t, dT, prm
