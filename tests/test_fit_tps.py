"""Tests for the classic TPS fit (M0). Phase 1 gate: noise-free analytic curves, k and alpha error < 0.5 %."""

import numpy as np
import pytest

from tpslib.config import load_sensor
from tpslib.dtau import PI15, dtau, tau_of_t
from tpslib.fit_tps import fit_tps
from tpslib.synth_analytic import make_case

SENSOR = load_sensor()
A, M = SENSOR.outer_radius, SENSOR.n_rings
GATE = 0.005

# 20 k values over 0.05-30 W/m/K, rho*cp cycling over 0.8-4.5 MJ/m^3/K
K_VALUES = np.geomspace(0.05, 30.0, 20)
RHO_CP = np.linspace(0.8e6, 4.5e6, 20)[np.argsort(np.random.default_rng(0).random(20))]
CASES = [(float(k), float(k / rc)) for k, rc in zip(K_VALUES, RHO_CP)]


@pytest.mark.parametrize("k,alpha", CASES)
def test_gate_constant_power(k, alpha):
    c = make_case(k, alpha, constant_current=False)
    r = fit_tps(c.t, c.dT, c.P, A, M)
    assert r.converged
    assert abs(r.k / k - 1) < GATE
    assert abs(r.alpha / alpha - 1) < GATE


@pytest.mark.parametrize("k,alpha", CASES)
def test_gate_constant_current(k, alpha):
    c = make_case(k, alpha, constant_current=True)
    r = fit_tps(c.t, c.dT, c.P_t, A, M)
    assert r.converged
    assert abs(r.k / k - 1) < GATE
    assert abs(r.alpha / alpha - 1) < GATE


def test_offset_B_recovered():
    k, alpha, P, B = 0.5, 3e-7, 0.05, 0.12
    t = np.arange(1, 4000) / 20.0
    dT = P / (PI15 * A * k) * dtau(tau_of_t(t, alpha, A), M) + B
    r = fit_tps(t, dT, P, A, M)
    assert r.k == pytest.approx(k, rel=1e-4)
    assert r.alpha == pytest.approx(alpha, rel=1e-4)
    # B also absorbs nothing else here, since data and fit share sigma_min
    assert r.B == pytest.approx(B, abs=1e-5)


def test_cutoff_independent():
    # Data generated with a different D(tau) cutoff -> only B changes, k and alpha do not.
    k, alpha, P = 2.0, 1e-6, 0.3
    t = np.arange(1, 20000) / 400.0
    t = t[alpha * t / A**2 <= 1.0]
    dT = P / (PI15 * A * k) * dtau(tau_of_t(t, alpha, A), M, sigma_min=1e-9)
    r = fit_tps(t, dT, P, A, M)
    assert r.k == pytest.approx(k, rel=1e-4)
    assert r.alpha == pytest.approx(alpha, rel=1e-4)


def test_window_truncation_flag():
    c = make_case(0.3, 2e-7, tau_end=0.9)
    r = fit_tps(c.t, c.dT, c.P, A, M)
    assert r.window_truncated
    assert abs(r.k / 0.3 - 1) < GATE
