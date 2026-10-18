"""Analytic Gustafsson D(tau) for an m-ring TPS sensor, plus the ideal temperature-rise model.

D(tau) = [m(m+1)]^-2 * int_{sigma_min}^{tau} dsigma / sigma^2
         * sum_l sum_k l k exp(-(l^2+k^2)/(4 m^2 sigma^2)) I0(l k / (2 m^2 sigma^2))

The product exp(-(l^2+k^2)/(4m^2s^2)) * I0(lk/(2m^2s^2)) is evaluated as
exp(-(l-k)^2/(4m^2s^2)) * i0e(lk/(2m^2s^2)) to avoid overflow.

The integrand behaves like 1/(2 sqrt(pi) (m+1) sigma) as sigma -> 0, so the integral diverges
logarithmically (infinitely thin rings). A lower limit sigma_min is therefore part of the
definition; it adds a constant ln(1/sigma_min)/(2 sqrt(pi)(m+1)) that the TPS fit absorbs into B.
The reference values in PROJECT_RULES.md correspond to sigma_min = 1e-6.

Integration is done in u = ln(sigma), where the integrand g(u) = sigma * f(sigma) is smooth and
tends to a constant as sigma -> 0.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import cumulative_simpson, quad
from scipy.interpolate import CubicSpline
from scipy.special import i0e

from tpslib.config import CONFIG_DIR, load_yaml

_CFG = load_yaml(CONFIG_DIR / "analysis.yaml")["dtau"]
SIGMA_MIN = float(_CFG["sigma_min"])
TABLE_TAU_MAX = float(_CFG["table_tau_max"])
TABLE_POINTS = int(_CFG["table_points"])

PI15 = np.pi**1.5


def _ring_sum(sigma: np.ndarray, m: int) -> np.ndarray:
    """sigma * f(sigma): the integrand in u = ln(sigma). Vectorised over sigma."""
    sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
    idx = np.arange(1, m + 1, dtype=float)
    lk = np.outer(idx, idx)                      # (m, m)
    dl2 = (idx[:, None] - idx[None, :]) ** 2     # (m, m)
    out = np.empty_like(sigma)
    chunk = 2048
    for s0 in range(0, sigma.size, chunk):
        s = sigma[s0 : s0 + chunk][:, None, None]
        q = 1.0 / (4.0 * m * m * s * s)
        terms = lk * np.exp(-dl2 * q) * i0e(2.0 * lk * q)
        out[s0 : s0 + chunk] = terms.sum(axis=(1, 2)) / s[:, 0, 0]
    return out / (m * (m + 1)) ** 2


def small_sigma_slope(m: int) -> float:
    """Limit of sigma*f(sigma) as sigma -> 0: 1 / (2 sqrt(pi) (m+1))."""
    return 1.0 / (2.0 * np.sqrt(np.pi) * (m + 1))


def dtau_exact(tau, m: int = 20, sigma_min: float = SIGMA_MIN) -> np.ndarray:
    """D(tau) by adaptive quadrature (slow, reference). D = 0 for tau <= sigma_min."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    out = np.zeros_like(tau)
    g = lambda u: _ring_sum(np.array([np.exp(u)]), m)[0]
    u0 = np.log(sigma_min)
    for i, t in enumerate(tau):
        if t <= sigma_min:
            continue
        u1 = np.log(t)
        # Breakpoints where the off-diagonal ring terms switch on (sigma ~ 1/(2m) .. 1)
        pts = [p for p in np.log([0.2 / m, 1.0 / m, 0.2, 1.0]) if u0 < p < u1]
        out[i] = quad(g, u0, u1, points=pts or None, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
    return out


@lru_cache(maxsize=8)
def _table(m: int, sigma_min: float, tau_max: float, n: int) -> CubicSpline:
    u = np.linspace(np.log(sigma_min), np.log(tau_max), n)
    d = cumulative_simpson(_ring_sum(np.exp(u), m), x=u, initial=0.0)
    return CubicSpline(u, d)


def dtau(tau, m: int = 20, sigma_min: float = SIGMA_MIN) -> np.ndarray:
    """Fast D(tau) from a precomputed table + cubic spline in ln(tau). D = 0 for tau <= sigma_min."""
    tau = np.asarray(tau, dtype=float)
    if np.any(tau > TABLE_TAU_MAX):
        raise ValueError(f"tau > {TABLE_TAU_MAX} outside the D(tau) table")
    spline = _table(m, sigma_min, TABLE_TAU_MAX, TABLE_POINTS)
    out = np.zeros_like(tau)
    pos = tau > sigma_min
    out[pos] = spline(np.log(tau[pos]))
    return out


def tau_of_t(t, alpha: float, a: float) -> np.ndarray:
    """Dimensionless time tau = sqrt(alpha t) / a (t < 0 -> 0)."""
    return np.sqrt(alpha * np.clip(np.asarray(t, dtype=float), 0.0, None)) / a


def ideal_rise_constant_power(t, k: float, alpha: float, a: float, P: float, m: int = 20) -> np.ndarray:
    """Mean sensor temperature rise for constant power: P / (pi^1.5 a k) * D(tau) [K]."""
    return P / (PI15 * a * k) * dtau(tau_of_t(t, alpha, a), m)


def ideal_rise_constant_current(
    t, k: float, alpha: float, a: float, I: float, R0: float, alpha_R: float, m: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Mean sensor temperature rise under constant current, P(t) = I^2 R0 (1 + alpha_R dT(t)).

    Duhamel superposition with power held piecewise constant between the given times
    (power on [t_{j}, t_{j+1}) is set by dT(t_j)). Returns (dT, P) at the given times.
    Accurate when the time grid is dense (e.g. log-spaced, >= ~1000 points).
    """
    t = np.asarray(t, dtype=float)
    if np.any(np.diff(t) <= 0) or t[0] <= 0:
        raise ValueError("t must be positive and strictly increasing")
    n = t.size
    steps_t = np.concatenate([[0.0], t[:-1]])   # step j switches on at steps_t[j]
    dP = np.zeros(n)
    P = np.zeros(n)
    dT = np.zeros(n)
    c = 1.0 / (PI15 * a * k)
    P_prev = 0.0
    for i in range(n):
        # Power during [steps_t[i], t[i]) is set by the temperature at steps_t[i]
        T_at_step = dT[i - 1] if i > 0 else 0.0
        P_now = I * I * R0 * (1.0 + alpha_R * T_at_step)
        dP[i] = P_now - P_prev
        P_prev = P_now
        lag = t[i] - steps_t[: i + 1]
        dT[i] = c * np.dot(dP[: i + 1], dtau(tau_of_t(lag, alpha, a), m))
        P[i] = I * I * R0 * (1.0 + alpha_R * dT[i])
    return dT, P
