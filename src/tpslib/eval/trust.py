"""Trust checks (plan 9.8, the parts that need no re-simulation), their thresholds and the post-hoc
variance scale.

Scores per measurement:
  epistemic   ensemble spread of ln k (members disagree -> unfamiliar input)
  total_std   total predictive std of ln k
  m0_m2       |ln k_M0 - ln k_M2|, NaN M0 (classic fit failed) counts as a flag
  m0_resid    M0 fit residual RMS / dT_max (the TPS model does not describe the curve)
  dT_range    dT_max outside [0.5, 8] K (plan 9.8)
v2 adds:
  epistemic_<t>  ensemble spread of every target t (ln k, ln alpha, ln G_c)
  range_<t>      predicted t outside the range the model produces on normal (val) measurements; this is
                 the "is the predicted G_c inside the training range" check of plan 9.8
v1 thresholds are per-check val quantiles (Phase 6). v2 thresholds use one shared tail fraction for all
continuous checks, chosen by bisection so that the combined false-alarm rate on val equals a budget.
Thresholds and scales come from the VALIDATION split only; the test splits are never used here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from tpslib.config import CONFIG_DIR, load_yaml

SCORES = ("epistemic", "total_std", "m0_m2", "m0_resid")
TARGET_NAMES = ("ln_k", "ln_alpha", "ln_G_c")


def load_trust_config() -> dict:
    return load_yaml(CONFIG_DIR / "trust.yaml")


def trust_scores(ens: dict, m0: dict) -> dict[str, np.ndarray]:
    ln_k = ens["mean"][:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        m0_m2 = np.abs(np.log(m0["k"]) - ln_k)
        resid = m0["residual_rms"] / np.abs(m0["dT_max"])
    s = {"epistemic": ens["std_epistemic"][:, 0], "total_std": ens["std"][:, 0],
         "m0_m2": m0_m2, "m0_resid": resid, "dT_max": m0["dT_max"]}
    for i, t in enumerate(TARGET_NAMES):
        s[f"pred_{t}"] = ens["mean"][:, i]
        s[f"epistemic_{t}"] = ens["std_epistemic"][:, i]
    return s


# --- v1 (Phase 6) ---------------------------------------------------------------------------------------

def fit_thresholds(scores_val: dict, q: float = 0.99) -> dict[str, float]:
    return {s: float(np.nanquantile(scores_val[s], q)) for s in SCORES}


def _dT_flag(scores: dict, dT_range) -> np.ndarray:
    return (scores["dT_max"] < dT_range[0]) | (scores["dT_max"] > dT_range[1])


def flags(scores: dict, thr: dict, dT_range=(0.5, 8.0)) -> dict[str, np.ndarray]:
    f = {}
    for s in SCORES:
        v = scores[s]
        f[s] = np.where(np.isnan(v), True, v > thr[s])
    f["dT_range"] = _dT_flag(scores, dT_range)
    f["any"] = np.any(np.stack([f[s] for s in (*SCORES, "dT_range")]), axis=0)
    return f


# --- v2: range checks, spread of every target, shared false-alarm budget -------------------------------

def _upper_scores(cfg: dict) -> list[str]:
    return ["total_std", "m0_m2", "m0_resid", *[f"epistemic_{t}" for t in cfg["epistemic_targets"]]]


def thresholds_at_tail(scores_val: dict, tail: float, cfg: dict) -> dict:
    """v2 thresholds for one shared tail fraction: upper checks at the (1 - tail) val quantile, each
    side of a range check at tail / 2."""
    return {"tail": float(tail),
            "upper": {s: float(np.nanquantile(scores_val[s], 1 - tail)) for s in _upper_scores(cfg)},
            "range": {t: [float(np.nanquantile(scores_val[f"pred_{t}"], tail / 2)),
                          float(np.nanquantile(scores_val[f"pred_{t}"], 1 - tail / 2))]
                      for t in cfg["range_targets"]}}


def flags_v2(scores: dict, thr: dict, dT_range=(0.5, 8.0)) -> dict[str, np.ndarray]:
    f = {}
    for s, v in thr["upper"].items():
        x = scores[s]
        f[s] = np.where(np.isnan(x), True, x > v)
    for t, (lo, hi) in thr["range"].items():
        x = scores[f"pred_{t}"]
        f[f"range_{t}"] = np.where(np.isnan(x), True, (x < lo) | (x > hi))
    f["dT_range"] = _dT_flag(scores, dT_range)
    f["any"] = np.any(np.stack(list(f.values())), axis=0)
    return f


def fit_thresholds_v2(scores_val: dict, cfg: dict | None = None, dT_range=None) -> dict:
    """Largest shared tail fraction whose combined val false-alarm rate stays within the budget."""
    cfg = cfg or load_trust_config()
    v2 = cfg["v2"]
    dT_range = dT_range or cfg["dT_range_K"]
    budget = float(v2["false_alarm_budget"])
    rate = lambda tail: float(flags_v2(scores_val, thresholds_at_tail(scores_val, tail, v2), dT_range)["any"].mean())
    lo, hi = (float(x) for x in v2["tail_search"])
    if rate(lo) > budget:  # checks that do not depend on the tail (NaN, dT range) already exceed it
        hi = lo
    for _ in range(60):
        if hi / lo < 1.0 + 1e-6:
            break
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if rate(mid) <= budget else (lo, mid)
    thr = thresholds_at_tail(scores_val, lo, v2)
    thr["val_false_alarm"] = rate(lo)
    thr["budget"] = budget
    return thr


# --- post-hoc variance scale ----------------------------------------------------------------------------

def fit_variance_scale(mean: np.ndarray, std: np.ndarray, true: np.ndarray, levels) -> np.ndarray:
    """Std multiplier per target so central intervals cover their nominal level (fit on val only).

    For each level l the exact multiplier is quantile_l(|err|/std) / z_l; the scale is the geometric
    mean over the levels (one number per target, applied as std * scale).
    """
    levels = np.asarray(levels, float)
    z = np.abs(mean - true) / std
    zn = norm.ppf(0.5 + levels / 2)
    ratios = np.nanquantile(z, levels, axis=0) / zn[:, None]
    return np.exp(np.mean(np.log(ratios), axis=0))


def interval_coverage(mean: np.ndarray, std: np.ndarray, true: np.ndarray, levels) -> np.ndarray:
    """Realised coverage of central Gaussian intervals, shape (n_levels, n_targets)."""
    z = np.abs(mean - true) / std
    zn = norm.ppf(0.5 + np.asarray(levels, float) / 2)
    return np.stack([np.mean(z <= c, axis=0) for c in zn])
