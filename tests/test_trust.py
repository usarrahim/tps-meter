"""Trust checks v2 (range checks, shared false-alarm budget) and the post-hoc variance scale."""

import numpy as np
import pytest

from tpslib.eval.trust import (fit_thresholds_v2, fit_variance_scale, flags_v2, interval_coverage,
                               load_trust_config, thresholds_at_tail, trust_scores)

CFG = load_trust_config()


def make_scores(rng, n, G_c_shift=0.0):
    """Synthetic trust scores of normal measurements (independent checks)."""
    s = {"total_std": rng.exponential(1, n), "m0_m2": rng.exponential(1, n), "m0_resid": rng.exponential(1, n),
         "dT_max": np.full(n, 2.5)}
    for t in ("ln_k", "ln_alpha", "ln_G_c"):
        s[f"epistemic_{t}"] = rng.exponential(1, n)
        s[f"pred_{t}"] = rng.normal(0, 1, n)
    s["pred_ln_G_c"] = s["pred_ln_G_c"] + G_c_shift
    return s


def test_v2_false_alarm_matches_budget():
    rng = np.random.default_rng(0)
    thr = fit_thresholds_v2(make_scores(rng, 20000), CFG)
    budget = CFG["v2"]["false_alarm_budget"]
    assert thr["val_false_alarm"] <= budget and thr["val_false_alarm"] > budget - 0.005
    fresh = flags_v2(make_scores(rng, 20000), thr, CFG["dT_range_K"])
    assert abs(fresh["any"].mean() - budget) < 0.01          # holds on unseen normal data


def test_v2_range_check_flags_out_of_range_G_c():
    rng = np.random.default_rng(1)
    thr = fit_thresholds_v2(make_scores(rng, 20000), CFG)
    lo, hi = thr["range"]["ln_G_c"]
    assert lo < -2.0 and hi > 2.0                              # about the 0.5 % / 99.5 % quantiles
    low = flags_v2(make_scores(rng, 2000, G_c_shift=-6.0), thr)
    assert low["range_ln_G_c"].mean() > 0.99 and low["any"].mean() > 0.99
    assert low["range_ln_k"].mean() < 0.02                     # the other targets stay in range


def test_v2_nan_and_dT_range_flagged():
    rng = np.random.default_rng(2)
    thr = fit_thresholds_v2(make_scores(rng, 5000), CFG)
    s = make_scores(rng, 10)
    s["m0_m2"][:] = np.nan                                     # failed classic fit
    assert flags_v2(s, thr)["m0_m2"].all()
    s = make_scores(rng, 10)
    s["dT_max"][:] = 12.0
    assert flags_v2(s, thr, CFG["dT_range_K"])["dT_range"].all()


def test_v2_budget_unreachable_gives_smallest_tail():
    rng = np.random.default_rng(3)
    s = make_scores(rng, 2000)
    s["m0_m2"][:200] = np.nan                                  # 10 % flagged whatever the threshold
    thr = fit_thresholds_v2(s, CFG)
    assert thr["tail"] == pytest.approx(CFG["v2"]["tail_search"][0])
    assert thr["val_false_alarm"] > CFG["v2"]["false_alarm_budget"]


def test_thresholds_at_tail_monotone():
    rng = np.random.default_rng(4)
    s = make_scores(rng, 5000)
    a, b = thresholds_at_tail(s, 0.01, CFG["v2"]), thresholds_at_tail(s, 0.05, CFG["v2"])
    assert all(a["upper"][k] > b["upper"][k] for k in a["upper"])
    assert all(a["range"][t][0] < b["range"][t][0] and a["range"][t][1] > b["range"][t][1] for t in a["range"])


def test_variance_scale_restores_nominal_coverage():
    rng = np.random.default_rng(5)
    n = 50000
    true = rng.normal(0, 1, (n, 3))
    sigma = np.array([0.01, 0.05, 0.02]) * rng.uniform(0.5, 2.0, (n, 1))
    mean = true + rng.normal(0, 1, (n, 3)) * sigma
    claimed = sigma * np.array([1.0, 2.0, 0.5])                # ok, too wide (like ln alpha), too narrow
    scale = fit_variance_scale(mean, claimed, true, CFG["variance_scale"]["levels"])
    assert scale == pytest.approx([1.0, 0.5, 2.0], rel=0.03)
    cov = interval_coverage(mean, claimed * scale, true, [0.68, 0.95])
    assert np.allclose(cov, [[0.68], [0.95]], atol=0.01)
    raw = interval_coverage(mean, claimed, true, [0.68])
    assert raw[0, 1] > 0.9 and raw[0, 2] < 0.4                  # the miscalibration the scale removes


def test_trust_scores_has_v2_keys():
    n = 4
    ens = {"mean": np.zeros((n, 3)) + [0.0, -14.0, 8.5], "std": np.full((n, 3), 0.02),
           "std_epistemic": np.full((n, 3), 0.005)}
    m0 = {"k": np.ones(n), "residual_rms": np.full(n, 1e-4), "dT_max": np.full(n, 2.0)}
    s = trust_scores(ens, m0)
    assert np.allclose(s["pred_ln_G_c"], 8.5) and np.allclose(s["epistemic_ln_alpha"], 0.005)
    assert np.allclose(s["m0_m2"], 0.0) and np.allclose(s["epistemic"], s["epistemic_ln_k"])
