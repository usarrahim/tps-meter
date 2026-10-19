"""M0+: full-physics inversion of one measurement with the r-z simulator (plan 9.8, docs/M0PLUS.md).

Starting from M2's estimate, the simulator is fitted to the measured curve:

  dT(t) = c S(t - t_c; k, alpha, h_c_bottom, h_c_top) + B0 + B1 t

S is the simulated mean sensor rise for the measured initial power (the response is exactly linear in
the power, the constant-current feedback included). The amplitude c absorbs the calibration errors that
scale the whole curve (alpha_R, power, radius; Gaussian prior around 1), t_c is the heating-start offset
(to first order, S(t - t_c) ~ S - t_c dS/dt), B0 an offset and B1 a residual linear drift; all four are
solved in closed form at every evaluation (variable projection). The four nonlinear parameters are fitted in
ln space with a trust-region least-squares solver and a forward-difference Jacobian.

Outputs used by the trust system:
  chi2_nu      weighted residual per degree of freedom: the simulator with the fitted parameters does not
               explain the curve (air gap below the prior, unmodelled physics, measurement fault)
  asymmetry    fitted h_max / h_min: the two contacts differ more than the training domain allows
The fitted k and alpha are the M0+ estimate; their difference to M2 is a check as well.

The film (layer thicknesses and materials) is a property of the sensor and is calibrated once on the
real device; the fit holds it fixed at the values it is given.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq, least_squares

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.data.features import _bin, time_grid
from tpslib.data.prior import FILM_LAYERS, contact_conductance
from tpslib.measurement.preprocess import Processed


def load_m0plus_config() -> dict:
    """Load configs/m0plus.yaml."""
    return load_yaml(CONFIG_DIR / "m0plus.yaml")


@dataclass
class M0PlusResult:
    """Fitted parameters and fit diagnostics of one measurement."""
    k: float
    alpha: float
    h_c_bottom: float
    h_c_top: float
    G_c: float
    asymmetry: float      # h_max / h_min
    chi2_nu: float
    scale: float          # fitted amplitude scale c (calibration errors)
    t_c: float            # fitted heating-start offset [s]
    B0: float             # fitted offset [K]
    B1: float             # fitted residual drift [K/s]
    n_bins: int
    nfev: int             # simulations used (residuals + Jacobian columns)
    time_s: float
    ok: bool
    message: str = ""

    def as_dict(self) -> dict:
        """Plain dict (for JSON / tables)."""
        return asdict(self)


def nominal_film(sensor_cfg: dict) -> dict:
    """Film description at the nominal sensor values (no thickness scaling)."""
    m = sensor_cfg["materials"]
    film = {f"scale_{n}": 1.0 for n in FILM_LAYERS}
    film.update(k_polyimide=float(m["polyimide"]["k"]), k_adhesive=float(m["adhesive"]["k"]))
    return film


def film_from_row(prm: dict) -> dict:
    """Film description of a synthetic sample (the 'calibrated sensor' of that measurement)."""
    film = {f"scale_{n}": float(prm[f"scale_{n}"]) for n in FILM_LAYERS}
    film.update(k_polyimide=float(prm["k_polyimide"]), k_adhesive=float(prm["k_adhesive"]))
    return film


def gc_of(h_bottom: float, h_top: float, film: dict, sensor_cfg: dict) -> float:
    """Total interface conductance G_c for given contacts and film (same definition as the prior)."""
    return float(contact_conductance({**film, "h_c_bottom": h_bottom, "h_c_top": h_top}, sensor_cfg))


def split_contact(G_c: float, log_ratio: float, film: dict, sensor_cfg: dict, h_bounds) -> tuple[float, float]:
    """Contacts (h_bottom, h_top) with ln(h_bottom/h_top) = log_ratio whose total conductance is G_c.

    If G_c is above what the film allows, the upper h bound is returned (perfect contact).
    """
    lo, hi = (float(x) for x in h_bounds)
    r = np.exp(log_ratio)
    f = lambda lh: gc_of(np.exp(lh) * np.sqrt(r), np.exp(lh) / np.sqrt(r), film, sensor_cfg) - G_c
    a, b = np.log(lo), np.log(hi)
    if f(b) < 0:
        lh = b
    elif f(a) > 0:
        lh = a
    else:
        lh = brentq(f, a, b, xtol=1e-6)
    hb, ht = np.exp(lh) * np.sqrt(r), np.exp(lh) / np.sqrt(r)
    return float(np.clip(hb, lo, hi)), float(np.clip(ht, lo, hi))


def white_noise_sigma(proc: Processed) -> float:
    """Per-sample white-noise level of dT from the second differences of the late half of the record.

    Second differences remove the smooth curve (and any linear drift); MAD keeps it robust to jumps.
    """
    y = proc.dT[proc.dT.size // 2:]
    if y.size < 8:
        return float("nan")
    d2 = np.diff(y, 2)
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def binned(proc: Processed, cfg: dict, alpha_R: float) -> dict:
    """Record on the feature log-time grid: populated bins after t_min, at their mean time.

    Returns t, dT, sigma per bin and the initial power P0 (power with the constant-current rise removed).
    """
    edges, _ = time_grid(proc.t_end)
    y, lt, cnt = _bin(proc.t, proc.dT, edges)
    keep = (cnt > 0) & (np.exp(np.nan_to_num(lt, nan=-np.inf)) >= float(cfg["t_min"]))
    t, y, cnt = np.exp(lt[keep]), y[keep], cnt[keep]
    s_w = white_noise_sigma(proc)
    s_w = s_w if np.isfinite(s_w) and s_w > 0 else 1e-6
    sigma = np.sqrt(s_w ** 2 / cnt + (float(cfg["sigma_rel_floor"]) * y) ** 2)
    P0 = float(np.median(proc.P / (1.0 + alpha_R * proc.dT)))
    return {"t": t, "dT": y, "sigma": sigma, "P0": P0, "sigma_white": s_w}


class _Model:
    """Simulated curve on the binned grid for a parameter vector x = ln(k, alpha, h_b, h_t)."""

    def __init__(self, proc: Processed, data: dict, film: dict, sensor_cfg: dict, numerics: dict,
                 scale_prior: float):
        from tpslib.sim.sensor_model import SimParams
        self.SimParams = SimParams
        self.proc, self.d, self.film = proc, data, film
        self.sensor_cfg, self.numerics = sensor_cfg, numerics
        self.n_sim = 0
        w = 1.0 / data["sigma"]
        self.w = w
        self.yw = data["dT"] * w
        self.basis = np.stack([np.ones_like(data["t"]), data["t"]], axis=1) * w[:, None]
        self.prior_w = 1.0 / float(scale_prior)   # Gaussian prior on the amplitude scale c around 1

    def curve(self, x: np.ndarray) -> np.ndarray:
        """Simulated dT on the data times."""
        from tpslib.sim.sensor_model import simulate
        k, alpha, hb, ht = np.exp(x)
        f = self.film
        sp = self.SimParams(
            k=float(k), rho_cp=float(k / alpha), P0=self.d["P0"], t_end=float(self.d["t"][-1]),
            h_c_bottom=float(hb), h_c_top=float(ht), sample_radius=self.proc.radius,
            thickness_bottom=self.proc.thickness, thickness_top=self.proc.thickness,
            k_polyimide=f["k_polyimide"], k_adhesive=f["k_adhesive"],
            layer_scale={n: f[f"scale_{n}"] for n in FILM_LAYERS}, constant_current=True,
            t_out_min=float(self.d["t"][0]))
        self.n_sim += 1
        return simulate(sp, sensor_cfg=self.sensor_cfg, numerics=self.numerics, t_out=self.d["t"]).dT

    def residual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted residual after the closed-form linear part; also returns (c, t_c, B0, B1).

        Linear part: amplitude scale c (absorbs the alpha_R, power and radius calibration errors, with a
        Gaussian prior of width `scale_prior` around 1, appended as one extra residual row), heating-start
        offset t_c (first order), offset B0 and residual drift B1.
        """
        s = self.curve(x)
        ds = -np.gradient(s, self.d["t"])          # S(t - t_c) ~ S(t) - t_c dS/dt (first order in t_c)
        A = np.vstack([np.column_stack([s * self.w, ds * self.w, self.basis]), [self.prior_w, 0.0, 0.0, 0.0]])
        rhs = np.append(self.yw, self.prior_w)
        B, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        return rhs - A @ B, B


def fit_m0plus(proc: Processed, start: dict, film: dict, cfg: dict | None = None,
               sensor_cfg: dict | None = None, numerics: dict | None = None) -> M0PlusResult:
    """Fit k, alpha and both contacts to one preprocessed record.

    start : {"k", "alpha", "G_c"} (typically the M2 estimate, physical units)
    film  : film description (`film_from_row` for synthetic data, `nominal_film` or a calibration otherwise)
    """
    from tpslib.sim.sensor_model import load_numerics, load_sensor_raw
    t0 = time.perf_counter()
    cfg = cfg or load_m0plus_config()
    fc = cfg["fit"]
    sensor_cfg = sensor_cfg or load_sensor_raw()
    numerics = numerics or load_numerics()
    alpha_R = float(sensor_cfg["electrical"]["alpha_R"])
    b = fc["bounds"]
    lo = np.log([b["k"][0], b["alpha"][0], b["h_c"][0], b["h_c"][0]])
    hi = np.log([b["k"][1], b["alpha"][1], b["h_c"][1], b["h_c"][1]])
    nan = float("nan")

    data = binned(proc, fc, alpha_R)
    n = data["t"].size
    if n < int(fc["min_bins"]):
        return M0PlusResult(nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, n, 0, time.perf_counter() - t0,
                            False, "too few bins")
    model = _Model(proc, data, film, sensor_cfg, numerics, float(fc["scale_prior_rel"]))
    step = float(fc["jac_step"])
    cache: dict = {}

    def fun(x):
        r, B = model.residual(x)
        cache["x"], cache["r"], cache["B"] = x.copy(), r, B
        return r

    def jac(x):
        r0 = cache["r"] if "x" in cache and np.array_equal(cache["x"], x) else model.residual(x)[0]
        J = np.empty((r0.size, x.size))
        for i in range(x.size):
            xi = x.copy()
            xi[i] += step if xi[i] + step <= hi[i] else -step
            J[:, i] = (model.residual(xi)[0] - r0) / (xi[i] - x[i])
        return J

    best = None
    for lr in fc["asym_starts"]:
        hb, ht = split_contact(float(start["G_c"]), float(lr), film, sensor_cfg, b["h_c"])
        x0 = np.clip(np.log([start["k"], start["alpha"], hb, ht]), lo + 1e-9, hi - 1e-9)
        try:
            sol = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", x_scale=1.0,
                                max_nfev=int(fc["max_nfev"]), ftol=float(fc["ftol"]), xtol=1e-4, gtol=1e-8)
        except Exception as exc:  # noqa: BLE001 - a failed inversion is itself a result
            if best is None:
                best = (np.inf, None, None, repr(exc)[:200])
            continue
        r, B = model.residual(sol.x) if not np.array_equal(cache.get("x"), sol.x) else (cache["r"], cache["B"])
        chi2 = float(np.sum(r * r))
        if best is None or chi2 < best[0]:
            best = (chi2, sol.x, B, sol.message)

    chi2, x, B, msg = best
    if x is None:
        return M0PlusResult(nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, n, model.n_sim,
                            time.perf_counter() - t0, False, msg)
    k, alpha, hb, ht = (float(v) for v in np.exp(x))
    return M0PlusResult(k=k, alpha=alpha, h_c_bottom=hb, h_c_top=ht, G_c=gc_of(hb, ht, film, sensor_cfg),
                        asymmetry=max(hb, ht) / min(hb, ht), chi2_nu=chi2 / max(n - 8, 1),
                        scale=float(B[0]), t_c=float(B[1] / B[0]), B0=float(B[2]), B1=float(B[3]), n_bins=n, nfev=model.n_sim,
                        time_s=time.perf_counter() - t0, ok=True, message=str(msg))


def m0plus_light(proc: Processed, start: dict, film: dict, cfg: dict | None = None,
                 sensor_cfg: dict | None = None, numerics: dict | None = None) -> dict:
    """Re-simulation at the M2 estimate (k, alpha, G_c) for a few contact splits: screening version of M0+.

    For every ln(h_bottom/h_top) in `checks.light_split_ratios` (G_c fixed at M2's value) one simulation is
    run and only the linear part (scale, t_c, offset, drift) is fitted. Returns the chi2_nu of every split,
    the smallest one and the split that gives it. A large minimum means no measurement close to M2's
    answer explains the curve; a best split far from 0 points to asymmetric contact.
    """
    from tpslib.sim.sensor_model import load_numerics, load_sensor_raw
    t0 = time.perf_counter()
    cfg = cfg or load_m0plus_config()
    fc = cfg["fit"]
    ratios = [float(r) for r in cfg["checks"]["light_split_ratios"]]
    sensor_cfg = sensor_cfg or load_sensor_raw()
    numerics = numerics or load_numerics()
    data = binned(proc, fc, float(sensor_cfg["electrical"]["alpha_R"]))
    n = data["t"].size
    chi2 = np.full(len(ratios), np.nan)
    out = {"chi2_nu": float("nan"), "chi2_by_split": chi2, "best_split": float("nan"), "n_bins": n, "ok": False}
    if n < int(fc["min_bins"]):
        out["time_s"] = time.perf_counter() - t0
        return out
    model = _Model(proc, data, film, sensor_cfg, numerics, float(fc["scale_prior_rel"]))
    for j, lr in enumerate(ratios):
        hb, ht = split_contact(float(start["G_c"]), lr, film, sensor_cfg, fc["bounds"]["h_c"])
        try:
            r, _ = model.residual(np.log([start["k"], start["alpha"], hb, ht]))
            chi2[j] = float(np.sum(r * r) / max(n - 8, 1))
        except Exception:  # noqa: BLE001 - a failed simulation stays NaN and is flagged downstream
            pass
    if np.isfinite(chi2).any():
        j = int(np.nanargmin(chi2))
        out.update(chi2_nu=float(chi2[j]), best_split=ratios[j], ok=True)
    out["time_s"] = time.perf_counter() - t0
    return out
