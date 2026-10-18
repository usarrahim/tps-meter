"""Tests for the analytic D(tau)."""

import numpy as np
import pytest

from tpslib.dtau import (
    SIGMA_MIN,
    _ring_sum,
    dtau,
    dtau_exact,
    ideal_rise_constant_current,
    ideal_rise_constant_power,
    small_sigma_slope,
)

# PROJECT_RULES.md reference values, m = 20
REFERENCE = {0.1: 0.20553, 0.3: 0.35460, 0.5: 0.46487, 0.75: 0.55840, 1.0: 0.61900, 1.5: 0.68912}


@pytest.mark.parametrize("tau,expected", REFERENCE.items())
def test_exact_matches_reference(tau, expected):
    assert dtau_exact(tau, m=20)[0] == pytest.approx(expected, rel=5e-4)


@pytest.mark.parametrize("tau,expected", REFERENCE.items())
def test_table_matches_reference(tau, expected):
    assert dtau(np.array([tau]), m=20)[0] == pytest.approx(expected, rel=5e-4)


def test_table_vs_exact_dense():
    tau = np.geomspace(2 * SIGMA_MIN, 10.0, 200)
    exact = dtau_exact(tau, m=20)
    assert np.max(np.abs(dtau(tau, m=20) - exact)) < 1e-9


def test_table_other_m():
    tau = np.array([0.05, 0.4, 1.0, 2.0])
    assert np.allclose(dtau(tau, m=12), dtau_exact(tau, m=12), atol=1e-9)


def test_monotonic_and_zero_below_cutoff():
    tau = np.geomspace(1e-7, 15, 2000)
    d = dtau(tau)
    assert np.all(d[tau <= SIGMA_MIN] == 0)
    assert np.all(np.diff(d[tau > SIGMA_MIN]) > 0)


def test_small_sigma_log_divergence():
    # Integrand in ln(sigma) tends to 1/(2 sqrt(pi)(m+1)): D grows like ln(tau) at small tau,
    # which is why sigma_min is part of the definition (see DECISIONS.md).
    g = _ring_sum(np.array([1e-4, 1e-5]), 20)
    assert np.allclose(g, small_sigma_slope(20), rtol=1e-5)
    # Changing the cutoff only shifts D by a constant
    tau = np.array([0.2, 0.5, 1.0])
    shift = dtau(tau, sigma_min=1e-8) - dtau(tau, sigma_min=1e-6)
    assert np.allclose(shift, small_sigma_slope(20) * np.log(100), rtol=1e-6)


def test_ideal_rise_formula():
    a, k, alpha, P = 6.4e-3, 0.19, 1.3e-7, 0.033
    t = np.array([10.0, 100.0, 315.0])
    tau = np.sqrt(alpha * t) / a
    expected = P / (np.pi**1.5 * a * k) * dtau(tau)
    assert np.allclose(ideal_rise_constant_power(t, k, alpha, a, P), expected)


def test_constant_current_raises_power():
    a, k, alpha, R0, aR = 6.4e-3, 1.2, 6e-7, 4.0, 0.0039
    I = np.sqrt(0.13 / R0)
    t = np.geomspace(1e-3, a * a / alpha, 1500)
    dT, P = ideal_rise_constant_current(t, k, alpha, a, I, R0, aR)
    dT0 = ideal_rise_constant_power(t, k, alpha, a, I * I * R0)
    assert np.allclose(P, I * I * R0 * (1 + aR * dT))
    assert np.all(dT[-10:] > dT0[-10:])                     # extra power -> extra rise
    # Relative extra rise is of order alpha_R * dT (~1 %)
    rel = dT[-1] / dT0[-1] - 1
    assert 0.2 * aR * dT[-1] < rel < aR * dT[-1]
