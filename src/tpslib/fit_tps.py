"""M0: classic TPS fit.

Model (linear in A, B for a given alpha):
    dT(t) = A * Dw(t; alpha) + B * P(t) / P0
    Dw(t; alpha) = sum_j (dP_j / P0) * D(tau(t - t_j))     (Duhamel, power history)
    k = P0 / (pi^1.5 * a * A)

For constant power Dw = D(tau) and the model is the textbook dT = A D(tau) + B.
The B * P(t)/P0 term carries the thin-layer temperature drop (proportional to power) and the
sigma_min constant of D, both of which scale with P(t) under constant current.

alpha is chosen as the value that makes dT most linear in Dw inside the window
tau2_window[0] < tau^2 < tau2_window[1]. Because the window itself depends on alpha, the search is:
coarse log scan -> (fix window, bounded 1-D minimisation in ln alpha -> update window) until stable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize_scalar

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.dtau import PI15, dtau, tau_of_t

_CFG = load_yaml(CONFIG_DIR / "analysis.yaml")["m0"]


@dataclass
class M0Result:
    """Result of the classic TPS fit: k, alpha and the fitted line dT = A Dw + B P/P0."""
    k: float                 # [W/m/K]
    alpha: float             # [m^2/s]
    A: float                 # [K]
    B: float                 # [K]
    P0: float                # [W] reference power (power at t = 0+)
    residual_rms: float      # [K] weighted RMS residual in the window
    n_bins: int              # number of time bins in the final window
    t_window: tuple          # (t_start, t_end) [s]
    tau_max: float           # tau at the last data point with the fitted alpha
    window_truncated: bool   # data ended before the upper window edge
    iterations: int
    converged: bool

    def as_dict(self) -> dict:
        """Plain dict (JSON friendly)."""
        return asdict(self)


def _log_bins(t, y, n_bins, weights=None):
    """Average (t, y) into log-spaced bins; returns bin means and sample counts."""
    edges = np.geomspace(t[0], t[-1] * (1 + 1e-12), n_bins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    tb = np.bincount(idx, weights=t, minlength=n_bins)
    yb = np.bincount(idx, weights=y, minlength=n_bins)
    keep = cnt > 0
    return tb[keep] / cnt[keep], yb[keep] / cnt[keep], cnt[keep]


class PowerHistory:
    """Piecewise-constant power history: step j of height dP_j switches on at ts_j."""

    def __init__(self, t, P, n_steps):
        if np.ndim(P) == 0:
            self.P0 = float(P)
            self.ts = np.array([0.0])
            self.dPn = np.array([1.0])
            self._t, self._Pn = None, None
            return
        P = np.asarray(P, dtype=float)
        self.P0 = float(P[0])
        edges = np.concatenate([[0.0], np.geomspace(t[0], t[-1], n_steps)[1:]])
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, edges.size - 1)
        cnt = np.bincount(idx, minlength=edges.size)
        sums = np.bincount(idx, weights=P, minlength=edges.size)
        level = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        # Fill empty segments with the previous level
        for j in range(level.size):
            if np.isnan(level[j]):
                level[j] = level[j - 1] if j > 0 else self.P0
        self.ts = edges
        self.dPn = np.diff(np.concatenate([[0.0], level])) / self.P0
        self._t, self._Pn = t, P / self.P0

    def p_norm(self, tq):
        """P(t)/P0 at query times."""
        if self._t is None:
            return np.ones_like(tq)
        return np.interp(tq, self._t, self._Pn)

    def d_weighted(self, tq, alpha, a, m):
        """Dw(t) = sum_j dPn_j D(tau(t - ts_j))."""
        if self.ts.size == 1:
            return dtau(tau_of_t(tq, alpha, a), m)
        lag = tq[:, None] - self.ts[None, :]
        return dtau(tau_of_t(lag, alpha, a), m) @ self.dPn


def _solve_ab(Dw, pn, y, w):
    X = np.column_stack([Dw, pn])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ coef
    return coef, float(np.sqrt(np.sum(w * r * r) / np.sum(w)))


def fit_tps(
    t,
    dT,
    P,
    a: float,
    m: int = 20,
    tau2_window=None,
    alpha_bounds=None,
) -> M0Result:
    """Classic TPS fit of a temperature-rise record.

    t  : sample times after the start of heating [s], strictly increasing, > 0
    dT : mean sensor temperature rise [K]
    P  : heating power [W], scalar (constant power) or array like t (e.g. constant current)
    a  : sensor outer radius [m]
    """
    t = np.asarray(t, dtype=float)
    dT = np.asarray(dT, dtype=float)
    keep = t > 0
    t, dT = t[keep], dT[keep]
    if np.ndim(P) > 0:
        P = np.asarray(P, dtype=float)[keep]
    w_lo, w_hi = tau2_window or _CFG["tau2_window"]
    lo, hi = alpha_bounds or _CFG["alpha_bounds"]
    min_bins = int(_CFG["min_window_bins"])

    tb, yb, wb = _log_bins(t, dT, int(_CFG["n_bins"]))
    hist = PowerHistory(t, P, int(_CFG["n_power_steps"]))
    pn = hist.p_norm(tb)

    def window(alpha):
        tau2 = alpha * tb / a**2
        return (tau2 > w_lo) & (tau2 < w_hi)

    def rms(ln_alpha, mask):
        alpha = np.exp(ln_alpha)
        Dw = hist.d_weighted(tb[mask], alpha, a, m)
        return _solve_ab(Dw, pn[mask], yb[mask], wb[mask])[1]

    # Coarse scan: residual RMS relative to the signal span inside each candidate window
    grid = np.linspace(np.log(lo), np.log(hi), int(_CFG["coarse_points"]))
    best, best_score = None, np.inf
    for g in grid:
        mask = window(np.exp(g))
        if mask.sum() < min_bins:
            continue
        span = np.ptp(yb[mask])
        if span <= 0:
            continue
        score = rms(g, mask) / span
        if score < best_score:
            best, best_score = g, score
    if best is None:
        raise ValueError("No alpha in alpha_bounds gives enough points inside the fit window")

    step = grid[1] - grid[0]
    ln_alpha = best
    mask = window(np.exp(ln_alpha))
    converged = False
    it = 0
    for it in range(1, int(_CFG["max_iterations"]) + 1):
        res = minimize_scalar(
            rms, bounds=(ln_alpha - 1.5 * step, ln_alpha + 1.5 * step), args=(mask,),
            method="bounded", options={"xatol": 1e-7},
        )
        ln_alpha = float(res.x)
        new_mask = window(np.exp(ln_alpha))
        if new_mask.sum() < min_bins:
            break
        if np.array_equal(new_mask, mask):
            converged = True
            break
        mask = new_mask
        step = max(step / 3.0, 1e-3)

    alpha = float(np.exp(ln_alpha))
    Dw = hist.d_weighted(tb[mask], alpha, a, m)
    (A, B), r = _solve_ab(Dw, pn[mask], yb[mask], wb[mask])
    tau_max = float(tau_of_t(t[-1], alpha, a))
    return M0Result(
        k=float(hist.P0 / (PI15 * a * A)),
        alpha=alpha,
        A=float(A),
        B=float(B),
        P0=hist.P0,
        residual_rms=r,
        n_bins=int(mask.sum()),
        t_window=(float(tb[mask][0]), float(tb[mask][-1])),
        tau_max=tau_max,
        window_truncated=bool(tau_max**2 < w_hi),
        iterations=it,
        converged=converged,
    )
