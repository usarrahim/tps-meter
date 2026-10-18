"""Error metrics for k / alpha predictions, broken down by parameter bands."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("k", "alpha", "G_c")
K_BANDS = [(0.05, 0.5), (0.5, 5.0), (5.0, 30.0)]


def rel_errors(pred: dict) -> dict[str, np.ndarray]:
    """Relative errors exp(ln_pred - ln_true) - 1 per target."""
    return {t: np.expm1(pred["mean"][:, i] - pred["true"][:, i]) for i, t in enumerate(TARGETS)}


def summarize(err: np.ndarray) -> dict:
    a = np.abs(err)
    return {"n": int(a.size), "med_pct": float(np.median(a) * 100) if a.size else np.nan,
            "p95_pct": float(np.percentile(a, 95) * 100) if a.size else np.nan,
            "bias_pct": float(np.median(err) * 100) if a.size else np.nan}


def coverage(pred: dict, i: int = 0, z: float = 1.96) -> float:
    return float(np.mean(np.abs(pred["mean"][:, i] - pred["true"][:, i]) <= z * pred["std"][:, i]))


def breakdown(pred: dict, params: dict, by: dict[str, tuple[np.ndarray, list]]) -> pd.DataFrame:
    """Median / p95 relative errors of k and alpha per group.

    by: {group name: (values per sample, list of (lo, hi) bins or list of category values)}
    """
    err = rel_errors(pred)
    rows = []
    for name, (vals, bins) in by.items():
        for b in bins:
            sel = (vals >= b[0]) & (vals < b[1]) if isinstance(b, tuple) else (vals == b)
            if not np.any(sel):
                continue
            ek, ea = summarize(err["k"][sel]), summarize(err["alpha"][sel])
            rows.append({"group": name, "bin": str(b), "n": ek["n"],
                         "k_med_pct": ek["med_pct"], "k_p95_pct": ek["p95_pct"], "k_bias_pct": ek["bias_pct"],
                         "a_med_pct": ea["med_pct"], "a_p95_pct": ea["p95_pct"]})
    return pd.DataFrame(rows)


def calibration_floor(params: dict, power_error_rel: float, n_mc: int = 200_000, seed: int = 0) -> dict:
    """k error that no model can remove: alpha_R and radius calibration errors (prior) and the power
    error (uniform +-power_error_rel). k_est / k = (1 + e_aR)(1 + e_P) / (1 + e_a)."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, params["k"].size, n_mc)
    e = ((1 + params["alpha_R_calib_rel"][i]) * (1 + rng.uniform(-power_error_rel, power_error_rel, n_mc))
         / (1 + params["radius_calib_rel"][i]) - 1)
    return {"k_med_pct": float(np.median(np.abs(e)) * 100), "k_p95_pct": float(np.percentile(np.abs(e), 95) * 100)}
