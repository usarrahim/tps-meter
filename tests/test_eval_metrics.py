"""Tests for evaluation metrics and the ensemble combination."""

import numpy as np
import pytest

from tpslib.eval.metrics import breakdown, calibration_floor, coverage, rel_errors, summarize
from tpslib.eval.predict import ensemble


def _pred(n=1000, sd=0.02, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, 3))
    return {"mean": true + rng.normal(0, sd, (n, 3)), "std": np.full((n, 3), sd), "true": true,
            "index": np.arange(n)}


def test_rel_errors_and_coverage():
    p = _pred()
    e = rel_errors(p)["k"]
    assert summarize(e)["med_pct"] == pytest.approx(0.6745 * 2.0, rel=0.1)
    assert coverage(p, 0, 1.96) == pytest.approx(0.95, abs=0.02)


def test_breakdown_groups():
    p = _pred()
    vals = np.arange(1000)
    t = breakdown(p, {}, {"half": (vals, [(0, 500), (500, 1000)]), "cat": (vals % 2, [0, 1])})
    assert list(t["n"]) == [500, 500, 500, 500]


def test_calibration_floor_matches_analytic():
    rng = np.random.default_rng(1)
    params = {"k": np.ones(50000), "alpha_R_calib_rel": rng.normal(0, 0.01, 50000),
              "radius_calib_rel": rng.normal(0, 0.005, 50000)}
    f = calibration_floor(params, power_error_rel=0.0)
    assert f["k_med_pct"] == pytest.approx(0.6745 * np.hypot(1.0, 0.5), rel=0.05)


def test_ensemble_variance_decomposition():
    a, b = _pred(seed=1), _pred(seed=1)
    b["mean"] = b["mean"] + 0.1
    e = ensemble([a, b])
    assert np.allclose(e["std_epistemic"], 0.05)
    assert np.allclose(e["std"] ** 2, 0.02 ** 2 + 0.05 ** 2)
