"""M0+ re-simulation inversion: contact split, noise estimate, and recovery of k / alpha / contacts."""

import os

import numpy as np
import pytest

from tpslib.data.generate import simulate_params
from tpslib.data.prior import row, sample_split
from tpslib.eval.m0plus import (_Model, binned, film_from_row, fit_m0plus, gc_of, load_m0plus_config,
                                m0plus_light, nominal_film, split_contact, white_noise_sigma)
from tpslib.measurement.acquisition import acquire
from tpslib.measurement.preprocess import Processed, preprocess
from tpslib.sim.sensor_model import load_numerics, load_sensor_raw

SENSOR = load_sensor_raw()
CFG = load_m0plus_config()


@pytest.mark.parametrize("log_ratio", [0.0, np.log(20.0), -np.log(5.0)])
def test_split_contact_reproduces_G_c_and_ratio(log_ratio):
    film = nominal_film(SENSOR)
    hb, ht = split_contact(3000.0, log_ratio, film, SENSOR, CFG["fit"]["bounds"]["h_c"])
    assert gc_of(hb, ht, film, SENSOR) == pytest.approx(3000.0, rel=1e-5)
    assert np.log(hb / ht) == pytest.approx(log_ratio, abs=1e-6)


def test_split_contact_saturates_above_film_limit():
    film = nominal_film(SENSOR)
    lo, hi = CFG["fit"]["bounds"]["h_c"]
    hb, ht = split_contact(1e7, 0.0, film, SENSOR, (lo, hi))
    assert hb == pytest.approx(hi) and ht == pytest.approx(hi)


def test_white_noise_sigma_ignores_smooth_curve_and_drift():
    rng = np.random.default_rng(0)
    t = np.linspace(0.01, 100, 20000)
    dT = 2.0 * np.log1p(t) + 1e-3 * t + 1e-3 * rng.standard_normal(t.size)
    proc = Processed(t=t, dT=dT, P=np.full(t.size, 0.1), a=6.4e-3, sps=200, current_range=0,
                     thickness=0.02, radius=0.025)
    assert white_noise_sigma(proc) == pytest.approx(1e-3, rel=0.05)


def _asymmetric_case():
    """Clean record (no measurement effects) of an asymmetric, mid-k sample, and its parameters."""
    p = sample_split(200, seed=11)
    asym = np.maximum(p["h_c_bottom"], p["h_c_top"]) / np.minimum(p["h_c_bottom"], p["h_c_top"])
    i = int(np.where((p["k"] > 0.5) & (p["k"] < 3) & (asym > 15) & (p["t_end"] < 60))[0][0])
    prm = row(p, i)
    sim = simulate_params(prm)
    return preprocess(acquire(sim["t"], sim["dT"], prm, np.random.default_rng(0), effects=False)), prm


def test_residual_vanishes_at_truth_and_light_check_sees_wrong_contact():
    """At the true parameters the re-simulation reproduces the clean record; a symmetric contact split
    with the same G_c (what M2 would assume) leaves a clearly larger residual."""
    proc, prm = _asymmetric_case()
    film = film_from_row(prm)
    data = binned(proc, CFG["fit"], float(SENSOR["electrical"]["alpha_R"]))
    model = _Model(proc, data, film, SENSOR, load_numerics(),
                   float(CFG["fit"]["scale_prior_rel"]))
    r, B = model.residual(np.log([prm["k"], prm["alpha"], prm["h_c_bottom"], prm["h_c_top"]]))
    assert np.sqrt(np.mean(r[:-1] ** 2)) < 0.05                 # well below one sigma per bin
    assert B[0] == pytest.approx(1.0, abs=1e-3)                  # no scale error in a clean record
    light = m0plus_light(proc, {"k": prm["k"], "alpha": prm["alpha"], "G_c": prm["G_c"]}, film)
    ratios = CFG["checks"]["light_split_ratios"]
    sym = light["chi2_by_split"][ratios.index(0.0)]
    assert light["ok"] and sym > 100 * max(np.mean(r * r), 1e-6)
    true_split = np.log(prm["h_c_bottom"] / prm["h_c_top"])
    assert np.sign(light["best_split"]) == np.sign(true_split) and abs(light["best_split"]) >= 1.5
    assert light["chi2_nu"] < sym


@pytest.mark.skipif(not os.environ.get("TPS_SLOW_TESTS"), reason="~4 min; set TPS_SLOW_TESTS=1")
def test_fit_recovers_parameters_and_asymmetry_on_clean_record():
    """Start from a symmetric contact split and a 10 % wrong k: the full inversion must find k, alpha
    and the asymmetry (three contact-split starts)."""
    proc, prm = _asymmetric_case()
    start = {"k": prm["k"] * 1.1, "alpha": prm["alpha"] * 0.9, "G_c": prm["G_c"]}
    cfg = {**CFG, "fit": {**CFG["fit"], "asym_starts": [0.0, 3.0, -3.0]}}
    r = fit_m0plus(proc, start, film_from_row(prm), cfg)
    assert r.ok
    assert r.k == pytest.approx(prm["k"], rel=0.01)
    assert r.alpha == pytest.approx(prm["alpha"], rel=0.03)
    assert r.asymmetry > 10.0
