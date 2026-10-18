"""Finite-volume assembly and implicit time integration.

Unknowns: temperature rise above ambient in the sample cells (near zone on fine columns, far zone
on coarse columns, per sample) and in the film nodes (fine column j < n_film, layer l).
System: C dT/dt = -K T + s P(t),  P(t) = P0 (1 + c_fb * w.T)
  C  : heat capacities (diagonal), K : conductance matrix incl. convection to ambient,
  s  : source distribution (sums to 1), w : sensor readout weights (sum to 1),
  c_fb = alpha_R for constant current, 0 for constant power.

Time stepping: implicit Euler with local Richardson extrapolation
  T_{n+1} = 2 * IE(dt/2, IE(dt/2, T_n)) - IE(dt, T_n)
(second order, L-stable). dt doubles every `steps_per_level` steps; A(dt) = C/dt + K, and
A(dt/2) of a level equals A(dt) of the previous level, so each level needs one new LU.
The rank-1 power feedback is handled exactly with Sherman-Morrison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from tpslib.sim.geometry import Mesh


@dataclass
class System:
    K: sp.csc_matrix
    C: np.ndarray           # [J/K]
    g_amb: np.ndarray       # conductance to ambient per unknown [W/K] (already in K's diagonal)
    s: np.ndarray           # source distribution
    w: np.ndarray           # readout weights
    sample_mask: np.ndarray # True for sample cells
    n: int


class _Builder:
    """Collects conductance links, capacities and ambient conductances."""

    def __init__(self):
        self.rows, self.cols, self.vals = [], [], []
        self.extra = []  # raw symmetric K entries (rows, cols, vals)
        self.n = 0

    def alloc(self, count: int) -> int:
        off = self.n
        self.n += count
        return off

    def link(self, i, j, G):
        i, j = np.atleast_1d(i), np.atleast_1d(j)
        G = np.broadcast_to(np.asarray(G, dtype=float), i.shape).ravel()
        i, j = i.ravel(), j.ravel()
        keep = G > 0
        self.rows.append(i[keep]); self.cols.append(j[keep]); self.vals.append(G[keep])

    def mortar(self, i, J0, J1, w1, G):
        """Couple node i to the linear interpolation (1-w1) T_J0 + w1 T_J1 with conductance G.

        Adds G v v^T with v = e_i - (1-w1) e_J0 - w1 e_J1: symmetric, positive semidefinite and
        conservative (weights sum to one).
        """
        nodes = [np.asarray(i), np.asarray(J0), np.asarray(J1)]
        coef = [np.ones_like(w1), -(1.0 - w1), -w1]
        rows, cols, vals = [], [], []
        for a in range(3):
            for b in range(3):
                rows.append(nodes[a]); cols.append(nodes[b]); vals.append(G * coef[a] * coef[b])
        self.extra.append((np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)))


def _block(bld: _Builder, C: list, g_amb: list, rf: np.ndarray, dz: np.ndarray, k: float, rcp: float,
           outer_flat: bool, h_conv: float):
    """Rectangular block of sample cells (radial faces rf, axial sizes dz). Returns idx(j, kz)."""
    nr, nz = rf.size - 1, dz.size
    off = bld.alloc(nr * nz)
    idx = lambda j, kz: off + np.asarray(j) * nz + np.asarray(kz)
    rc = 0.5 * (rf[:-1] + rf[1:])
    A = np.pi * (rf[1:] ** 2 - rf[:-1] ** 2)
    J, Z = np.meshgrid(np.arange(nr), np.arange(nz), indexing="ij")
    C.append((idx(J, Z).ravel(), (rcp * A[J] * dz[Z]).ravel()))
    if nr > 1:
        J2, Z2 = J[:-1], Z[:-1]
        rface = rf[J2 + 1]
        bld.link(idx(J2, Z2), idx(J2 + 1, Z2), 2 * np.pi * rface * dz[Z2] * k / (rc[J2 + 1] - rc[J2]))
    if nz > 1:
        J2, Z2 = J[:, :-1], Z[:, :-1]
        bld.link(idx(J2, Z2), idx(J2, Z2 + 1), A[J2] * 2 * k / (dz[Z2] + dz[Z2 + 1]))
    if h_conv > 0:
        side = idx(np.full(nz, nr - 1), np.arange(nz))
        g_amb.append((side, 2 * np.pi * rf[-1] * dz / (1.0 / h_conv + (rf[-1] - rc[-1]) / k)))
        if outer_flat:
            g_amb.append((idx(np.arange(nr), nz - 1), A / (1.0 / h_conv + dz[-1] / (2 * k))))
    return idx


def assemble(mesh: Mesh, k_bot: float, rc_bot: float, k_top: float, rc_top: float,
             h_bot: float, h_top: float, h_conv: float, k_air: float) -> System:
    bld = _Builder()
    C_parts, amb_parts = [], []
    rf, rfc, A = mesh.r_faces, mesh.r_faces_coarse, mesh.annulus

    near_idx, sample_ranges = {}, []
    for side, sm, k, rcp in (("b", mesh.bottom, k_bot, rc_bot), ("t", mesh.top, k_top, rc_top)):
        dz = sm.dz
        start = bld.n
        has_far = sm.n_near < dz.size
        near = _block(bld, C_parts, amb_parts, rf, dz[: sm.n_near], k, rcp, not has_far, h_conv)
        if has_far:
            far = _block(bld, C_parts, amb_parts, rfc, dz[sm.n_near:], k, rcp, True, h_conv)
            jn = np.arange(mesh.nr)
            G = A / ((dz[sm.n_near - 1] + dz[sm.n_near]) / (2 * k))
            # Far-zone temperature at each fine column: linear interpolation between coarse centers
            rcc = 0.5 * (rfc[:-1] + rfc[1:])
            pos = np.clip(np.interp(mesh.r_c, rcc, np.arange(rcc.size)), 0, rcc.size - 1)
            J0 = np.minimum(np.floor(pos).astype(int), rcc.size - 2) if rcc.size > 1 else np.zeros(mesh.nr, int)
            w1 = pos - J0 if rcc.size > 1 else np.zeros(mesh.nr)
            J1 = np.minimum(J0 + 1, rcc.size - 1)
            bld.mortar(near(jn, sm.n_near - 1), far(J0, 0), far(J1, 0), w1, G)
        near_idx[side] = near
        sample_ranges.append((start, bld.n))

    nf, layers = mesh.n_film, mesh.layers
    nl = len(layers)
    of = bld.alloc(nf * nl)
    idx_f = lambda j, l: of + np.asarray(j) * nl + l
    jf = np.arange(nf)
    rc = mesh.r_c
    dzb0, dzt0 = mesh.bottom.dz[0], mesh.top.dz[0]
    inv = lambda h: 0.0 if np.isinf(h) else 1.0 / h
    for l, L in enumerate(layers):
        C_parts.append((idx_f(jf, l), L.rho_cp * A[jf] * L.thickness))
        if L.thickness > 0 and nf > 1:
            j0 = jf[:-1]
            rface = rf[j0 + 1]
            R = (rface - rc[j0]) / L.k[j0] + (rc[j0 + 1] - rface) / L.k[j0 + 1]
            bld.link(idx_f(j0, l), idx_f(j0 + 1, l), 2 * np.pi * rface * L.thickness / R)
    half = lambda L: (L.thickness / (2 * L.k[jf])) if L.thickness > 0 else np.zeros(nf)
    bld.link(near_idx["b"](jf, 0), idx_f(jf, 0), A[jf] / (dzb0 / (2 * k_bot) + inv(h_bot) + half(layers[0])))
    for l in range(nl - 1):
        bld.link(idx_f(jf, l), idx_f(jf, l + 1), A[jf] / (half(layers[l]) + half(layers[l + 1])))
    bld.link(idx_f(jf, nl - 1), near_idx["t"](jf, 0), A[jf] / (half(layers[-1]) + inv(h_top) + dzt0 / (2 * k_top)))
    # outside the film: air gap of the film thickness between the samples
    t_film = sum(L.thickness for L in layers)
    ja = np.arange(nf, mesh.nr)
    if ja.size:
        bld.link(near_idx["b"](ja, 0), near_idx["t"](ja, 0),
                 A[ja] / (dzb0 / (2 * k_bot) + t_film / k_air + dzt0 / (2 * k_top)))

    n = bld.n
    C = np.zeros(n)
    for i, v in C_parts:
        C[i] = v
    g_amb = np.zeros(n)
    for i, v in amb_parts:
        np.add.at(g_amb, i, v)
    r_all = np.concatenate(bld.rows); c_all = np.concatenate(bld.cols); v_all = np.concatenate(bld.vals)
    off = sp.coo_matrix((-v_all, (r_all, c_all)), shape=(n, n))
    diag = np.bincount(r_all, weights=v_all, minlength=n) + np.bincount(c_all, weights=v_all, minlength=n)
    K = off + off.T + sp.diags(diag + g_amb)
    for er, ec, ev in bld.extra:
        K = K + sp.coo_matrix((ev, (er, ec)), shape=(n, n))
    K = K.tocsc()

    heater = [l for l, L in enumerate(layers) if L.heater][0]
    trace_cols = np.where(mesh.is_trace[:nf])[0]
    s = np.zeros(n)
    # Uniform heat per trace volume <=> per trace area (same copper cross-section on every ring),
    # i.e. ring power proportional to ring length. The readout uses the same weighting.
    s[idx_f(trace_cols, heater)] = A[trace_cols]
    s /= s.sum()
    sample_mask = np.zeros(n, bool)
    for a0, a1 in sample_ranges:
        sample_mask[a0:a1] = True
    return System(K=K, C=C, g_amb=g_amb, s=s, w=s.copy(), sample_mask=sample_mask, n=n)


@dataclass
class Integration:
    t: np.ndarray           # step times [s]
    Tbar: np.ndarray        # sensor reading at step times [K]
    P: np.ndarray           # power at step times [W]
    T_final: np.ndarray
    E_in: float             # [J]
    E_loss: float           # [J]
    n_factorizations: int


def _factor(K, C, coef):
    # Symmetric (SPD) system: minimum degree on A^T+A gives ~40 % less fill than COLAMD.
    return splu((K + sp.diags(C * coef)).tocsc(), permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})


def integrate_trbdf2(sys: System, P0: float, c_fb: float, t_end: float, dt0: float,
                     steps_per_level: int) -> Integration:
    """TR-BDF2 with gamma = 2 - sqrt(2) (L-stable, 2nd order).

    For this gamma both stages use the same matrix mu C/dt + K (mu = 2/gamma), so one LU per level
    and two back-substitutions per step (vs three for extrapolated implicit Euler). Solves are
    memory-bandwidth bound, which dominates when many simulations run in parallel.
    Energy is booked with the scheme's own quadrature, so the discrete balance is exact.
    """
    C, K, s, w, g = sys.C, sys.K, sys.s, sys.w, sys.g_amb
    gam = 2.0 - np.sqrt(2.0)
    mu = 2.0 / gam
    c_b = (1.0 - gam) / (2.0 - gam)                  # BDF2 stage: T1 - c_b dt f(T1) = a T_g + b T_n
    a_b = 1.0 / (gam * (2.0 - gam))
    b_b = -((1.0 - gam) ** 2) / (gam * (2.0 - gam))
    cf = P0 * c_fb
    cache: dict[int, tuple] = {}

    def factor(level: int):
        if level not in cache:
            lu = _factor(K, C, mu / (dt0 * 2.0**level))
            y = lu.solve(s)
            cache[level] = (lu, y, float(w @ y))
        return cache[level]

    def solve_fb(rhs, level):
        """Solve A T = rhs + s P0 (1 + c_fb w.T) exactly (Sherman-Morrison)."""
        lu, y, wy = factor(level)
        x = lu.solve(rhs + s * P0)
        return x + y * (cf * (w @ x) / (1.0 - cf * wy)) if c_fb else x

    # Start-up: the power step at t = 0 excites very stiff modes that the trapezoidal stage does not
    # damp, so the first level is done with extrapolated implicit Euler (strongly damping).
    first = integrate(sys, P0, c_fb, min(t_end, dt0 * steps_per_level), dt0, steps_per_level, max_levels=1)
    T = first.T_final
    P_n = float(first.P[-1])
    t, level = float(first.t[-1]), 1
    ts, Tb, Ps = list(first.t), list(first.Tbar), list(first.P)
    E_in, E_loss = first.E_in, first.E_loss
    n_fact0 = first.n_factorizations
    while t < t_end * (1 - 1e-12):
        dt = dt0 * 2.0**level
        mc = C * (mu / dt)
        for _ in range(steps_per_level):
            # TR stage to t + gamma dt
            T_g = solve_fb(mc * T - K @ T + s * P_n, level)
            P_g = P0 * (1.0 + c_fb * (w @ T_g))
            e_tr = 0.5 * gam * dt * (P_n + P_g)
            l_tr = 0.5 * gam * dt * (g @ (T + T_g))
            # BDF2 stage to t + dt
            T_new = solve_fb(mc * (a_b * T_g + b_b * T), level)
            P_new = P0 * (1.0 + c_fb * (w @ T_new))
            E_in += a_b * e_tr + c_b * dt * P_new
            E_loss += a_b * l_tr + c_b * dt * (g @ T_new)
            T, P_n = T_new, P_new
            t += dt
            ts.append(t); Tb.append(float(w @ T)); Ps.append(P_n)
            if t >= t_end * (1 - 1e-12):
                break
        level += 1
    return Integration(np.array(ts), np.array(Tb), np.array(Ps), T, E_in, E_loss, len(cache) + n_fact0)


def integrate(sys: System, P0: float, c_fb: float, t_end: float, dt0: float, steps_per_level: int,
              max_levels: int | None = None) -> Integration:
    """Implicit Euler with local Richardson extrapolation (3 solves per step)."""
    C, K, s, w, g = sys.C, sys.K, sys.s, sys.w, sys.g_amb
    cache: dict[int, tuple] = {}

    def factor(level: int):
        if level not in cache:
            dt = dt0 * 2.0**level
            # Symmetric (SPD) system: minimum degree on A^T+A gives ~40 % less fill than COLAMD.
            lu = splu((K + sp.diags(C / dt)).tocsc(), permc_spec="MMD_AT_PLUS_A",
                      options={"SymmetricMode": True})
            y = lu.solve(s)
            cache[level] = (lu, y, float(w @ y))
        return cache[level]

    def ie_step(T, dt, level):
        lu, y, wy = factor(level)
        x = lu.solve(C / dt * T + s * P0)
        if c_fb:
            cf = P0 * c_fb
            T_new = x + y * (cf * (w @ x) / (1.0 - cf * wy))
        else:
            T_new = x
        P = P0 * (1.0 + c_fb * (w @ T_new))
        return T_new, P * dt, (g @ T_new) * dt

    T = np.zeros(sys.n)
    t, level = 0.0, 0
    ts, Tb, Ps = [], [], []
    E_in = E_loss = 0.0
    while t < t_end * (1 - 1e-12):
        dt = dt0 * 2.0**level
        for _ in range(steps_per_level):
            Tf, ef, lf = ie_step(T, dt, level)
            Th, eh1, lh1 = ie_step(T, dt / 2, level - 1)
            Th, eh2, lh2 = ie_step(Th, dt / 2, level - 1)
            T = 2.0 * Th - Tf
            E_in += 2.0 * (eh1 + eh2) - ef
            E_loss += 2.0 * (lh1 + lh2) - lf
            t += dt
            tb = float(w @ T)
            ts.append(t); Tb.append(tb); Ps.append(P0 * (1.0 + c_fb * tb))
            if t >= t_end * (1 - 1e-12):
                break
        level += 1
        if max_levels is not None and level >= max_levels:
            break
    return Integration(np.array(ts), np.array(Tb), np.array(Ps), T, E_in, E_loss, len(cache))
