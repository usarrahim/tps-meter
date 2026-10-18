"""Simple trust checks (plan 9.8, the parts that need no re-simulation) and their thresholds.

Scores per measurement:
  epistemic   ensemble spread of ln k (members disagree -> unfamiliar input)
  total_std   total predictive std of ln k
  m0_m2       |ln k_M0 - ln k_M2|, NaN M0 (classic fit failed) counts as a flag
  m0_resid    M0 fit residual RMS / dT_max (the TPS model does not describe the curve)
  dT_range    dT_max outside [0.5, 8] K (plan 9.8)
Thresholds for the continuous scores are quantiles of the VALIDATION split (normal measurements), so
the false-alarm rate on normal data is about (1 - q) per check; the test splits are never used here.
"""

from __future__ import annotations

import numpy as np

SCORES = ("epistemic", "total_std", "m0_m2", "m0_resid")


def trust_scores(ens: dict, m0: dict) -> dict[str, np.ndarray]:
    ln_k = ens["mean"][:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        m0_m2 = np.abs(np.log(m0["k"]) - ln_k)
        resid = m0["residual_rms"] / np.abs(m0["dT_max"])
    return {"epistemic": ens["std_epistemic"][:, 0], "total_std": ens["std"][:, 0],
            "m0_m2": m0_m2, "m0_resid": resid, "dT_max": m0["dT_max"]}


def fit_thresholds(scores_val: dict, q: float = 0.99) -> dict[str, float]:
    return {s: float(np.nanquantile(scores_val[s], q)) for s in SCORES}


def flags(scores: dict, thr: dict, dT_range=(0.5, 8.0)) -> dict[str, np.ndarray]:
    f = {}
    for s in SCORES:
        v = scores[s]
        f[s] = np.where(np.isnan(v), True, v > thr[s])
    f["dT_range"] = (scores["dT_max"] < dT_range[0]) | (scores["dT_max"] > dT_range[1])
    f["any"] = np.any(np.stack([f[s] for s in (*SCORES, "dT_range")]), axis=0)
    return f
