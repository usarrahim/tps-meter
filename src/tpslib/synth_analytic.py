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
    k: float
    alpha: float
    P: float          # initial power [W]
    sps: int
    t: np.ndarray     # [s]
    dT: np.ndarray    # [K]
    P_t: np.ndarray   # power at each sample [W]


def setup_power_and_rate(k, alpha, a, m, tau_end=1.0):
    P = float(_DEV["target_temperature_rise"]) * PI15 * a * k / float(dtau(np.array(1.0), m))
    P = float(np.clip(P, _DEV["power"]["min"], _DEV["power"]["max"]))
    t_end = tau_end**2 * a * a / alpha
    cap = int(_DEV["sample_rate"]["max_samples_per_record"])
    allowed = sorted(_DEV["sample_rate"]["allowed"])
    sps = max([r for r in allowed if r * t_end <= cap], default=allowed[0])
    return P, t_end, sps


def make_case(k, alpha, constant_current=False, tau_end=1.0, sensor=None) -> AnalyticCase:
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
