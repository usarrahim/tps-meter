"""Analytic reference for the simulator's analytic-limit gate: finite-width ring bands.

m annular bands [r_i - b/2, r_i + b/2] on the plane z = 0 of an infinite medium, uniform heat flux
over the band area (total power P, constant), perfect contact, no film. The area-averaged band
temperature is (Hankel transform in r, plane source in z):

    T_mean(t) = pi P / (k A^2) * int_0^inf W(lam)^2 erf(lam sqrt(alpha t)) dlam
    W(lam)    = sum_i [ r J1(lam r) / lam ]_{r_i - b/2}^{r_i + b/2},   A = sum_i 2 pi r_i b

Unlike D(tau) (infinitely thin rings) this is finite for all t, so it can be compared with the
simulator point by point. As b -> 0 it tends to P/(pi^1.5 a k) D(tau) + const.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.special import erf, j1


def band_weight(lam: np.ndarray, radii: np.ndarray, b: float) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    out = np.zeros_like(lam)
    small = lam < 1e-12
    for r in radii:
        for edge, sign in ((r + b / 2, 1.0), (r - b / 2, -1.0)):
            out[~small] += sign * edge * j1(lam[~small] * edge) / lam[~small]
            out[small] += sign * edge**2 / 2
    return out


def band_reference(t, k: float, alpha: float, P: float, a: float, m: int, b: float,
                   lam_max_factor: float = 400.0, points_per_period: int = 24) -> np.ndarray:
    """Mean band temperature rise [K] at times t for constant power P."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    radii = np.arange(1, m + 1) * a / m
    A = np.sum(2 * np.pi * radii * b)
    lam_max = lam_max_factor / b
    dlam = 2 * np.pi / (2 * (a + b / 2)) / points_per_period
    n = int(np.ceil(lam_max / dlam)) | 1
    lam = np.linspace(0.0, lam_max, n)
    W2 = band_weight(lam, radii, b) ** 2
    # Tail beyond lam_max: J1(x)^2 ~ 1/(pi x) on average, cross terms average out
    edges = np.concatenate([radii + b / 2, radii - b / 2])
    tail = np.sum(edges) / (2 * np.pi * lam_max**2)
    out = np.empty_like(t)
    for i, ti in enumerate(t):
        out[i] = simpson(W2 * erf(lam * np.sqrt(alpha * ti)), x=lam) + tail
    return np.pi * P / (k * A * A) * out
