"""Simulator tests: geometry, conservation, analytic limit, feedback and physical trends."""

import copy

import numpy as np
import pytest

from tpslib.config import load_sensor
from tpslib.fit_tps import fit_tps
from tpslib.sim.analytic import band_reference
from tpslib.sim.geometry import build_mesh, coarsen_faces, edge_graded, geometric_segment, radial_faces
from tpslib.sim.sensor_model import SimParams, load_numerics, load_sensor_raw, simulate
from tpslib.sim.solver import assemble

S = load_sensor()
A, M, B = S.outer_radius, S.n_rings, S.trace_width
NU = load_numerics()

# Coarse numerics for fast trend tests (trends do not need the production mesh)
FAST = copy.deepcopy(NU)
FAST["mesh"].update(edge_cell=0.0, n_gap=2, n_trace=1, r_growth=1.15, z_growth=1.25, z_growth_far=1.15,
                    coarse_growth=1.15)
FAST["time"]["steps_per_level"] = 4
FAST["output"]["n_points"] = 200

GLASS = dict(k=1.2, rho_cp=2.0e6)


def glass(**kw):
    p = dict(GLASS, P0=0.13, t_end=A * A / (1.2 / 2.0e6), h_c_bottom=5000.0, h_c_top=5000.0)
    p.update(kw)
    return SimParams(**p)


# --- geometry -------------------------------------------------------------------------------

def test_geometric_segment_exact_ends():
    f = geometric_segment(1.0, 3.0, 0.01, 1.2)
    assert f[0] == 1.0 and f[-1] == 3.0 and np.all(np.diff(f) > 0)
    d = np.diff(f)
    assert np.allclose(d[1:] / d[:-1], d[1] / d[0])


def test_edge_graded_symmetric():
    f = edge_graded(0.0, 1e-4, 2e-6, 1.5, 2)
    d = np.diff(f)
    assert d[0] == pytest.approx(d[-1]) and d[0] < 3e-6
    assert np.allclose(d, d[::-1])


def test_traces_aligned_with_rings():
    nm = NU["mesh"]
    f, ring = radial_faces(A, M, B, 10e-3, 25e-3, nm["n_center"], nm["n_trace"], nm["n_gap"],
                           nm["r_growth"], nm["edge_cell"], nm["edge_growth"])
    rc = 0.5 * (f[:-1] + f[1:])
    w = np.diff(f)
    for i in range(M):
        sel = ring == i
        r_i = (i + 1) * A / M
        assert np.sum(w[sel]) == pytest.approx(B, rel=1e-9)            # trace width b
        assert np.sum(rc[sel] * w[sel]) / np.sum(w[sel]) == pytest.approx(r_i, rel=1e-9)
    assert f[-1] == pytest.approx(25e-3)


def test_coarse_faces_subset():
    f, _ = radial_faces(A, M, B, 10e-3, 25e-3, 2, 1, 2, 1.05, 1.5e-6, 1.4)
    c = coarsen_faces(f, 1e-4, 1.03, A)
    assert np.all(np.isin(c, f)) and c[0] == 0 and c[-1] == f[-1]


def test_mesh_layers_and_heater():
    mesh = build_mesh(load_sensor_raw(), NU, 25e-3, 20e-3, 15e-3)
    names = [L.name for L in mesh.layers]
    assert names == ["polyimide_base", "copper_level", "adhesive", "polyimide_cover"]
    cu = mesh.layers[1]
    trace = mesh.is_trace[: mesh.n_film]
    assert np.all(cu.k[trace] == 398.0) and np.all(cu.k[~trace] == 0.20)
    assert mesh.r_c[mesh.n_film - 1] < 10e-3 <= mesh.r_c[mesh.n_film]


# --- assembly / conservation ----------------------------------------------------------------

def test_conductance_matrix_conserves_heat():
    mesh = build_mesh(load_sensor_raw(), NU, 25e-3, 20e-3, 20e-3)
    sys = assemble(mesh, 1.2, 2e6, 1.2, 2e6, 3000.0, 7000.0, 10.0, 0.026)
    # Internal links exchange heat only: K @ 1 = conductance to ambient
    assert np.allclose(sys.K @ np.ones(sys.n), sys.g_amb, atol=1e-12 * sys.K.diagonal().max())
    assert abs(sys.K - sys.K.T).max() < 1e-12 * sys.K.diagonal().max()
    assert sys.s.sum() == pytest.approx(1.0) and np.all(sys.C >= 0)


def test_energy_balance_discrete():
    r = simulate(glass(), numerics=FAST)
    assert abs(r.energy["rel_imbalance"]) < 1e-9
    assert 0.9 < r.energy["frac_stored_samples"] < 1.0


def test_constant_current_feedback():
    r = simulate(glass(constant_current=True), numerics=FAST)
    assert np.allclose(r.P, 0.13 * (1 + S.alpha_R * r.dT))
    r0 = simulate(glass(constant_current=False), numerics=FAST)
    assert np.all(r.dT[-5:] > r0.dT[-5:])


# --- analytic limit (production mesh) -------------------------------------------------------

def test_analytic_limit_gate():
    g = NU["analytic_limit"]
    p = glass(layers=False, h_c_bottom=np.inf, h_c_top=np.inf, constant_current=False, h_convection=0.0,
              sample_radius=g["sample_radius"], thickness_bottom=g["sample_thickness"],
              thickness_top=g["sample_thickness"])
    r = simulate(p)
    an = band_reference(r.t, p.k, p.alpha, p.P0, A, M, B)
    tau = np.sqrt(p.alpha * r.t) / A
    win = (tau >= 0.1) & (tau <= 1.0)
    assert np.abs(r.dT[win] / an[win] - 1).max() < 0.005
    f = fit_tps(r.t, r.dT, p.P0, A, M)
    assert abs(f.k / p.k - 1) < 0.005 and abs(f.alpha / p.alpha - 1) < 0.005


def test_band_reference_tends_to_dtau():
    from tpslib.dtau import PI15, dtau
    k, al, P = 1.2, 6e-7, 0.1
    t = np.geomspace(0.05, A * A / al, 8)
    tau = np.sqrt(al * t) / A
    diff = band_reference(t, k, al, P, A, M, 1e-6, lam_max_factor=40) - P / (PI15 * A * k) * dtau(tau)
    assert np.ptp(diff) < 2e-3 * P / (PI15 * A * k)   # only a constant offset remains


# --- physical trends (fast mesh) ------------------------------------------------------------

def test_lower_contact_raises_temperature_and_offset():
    hi = simulate(glass(h_c_bottom=20000.0, h_c_top=20000.0), numerics=FAST)
    lo = simulate(glass(h_c_bottom=1000.0, h_c_top=1000.0), numerics=FAST)
    assert np.all(lo.dT > hi.dT)


def test_thin_sample_deviates_late():
    thick = simulate(glass(), numerics=FAST)
    thin = simulate(glass(thickness_bottom=6e-3, thickness_top=6e-3), numerics=FAST)
    tau = np.sqrt(GLASS["k"] / GLASS["rho_cp"] * thick.t) / A
    dev = thin.dT / thick.dT - 1
    assert np.abs(dev[tau < 0.2]).max() < 1e-3      # early: identical
    assert dev[-1] > 5e-3                           # late: heat reflected by the adiabatic-ish far face


def test_top_bottom_asymmetry():
    a = simulate(glass(h_c_bottom=1000.0, h_c_top=20000.0), numerics=FAST)
    b = simulate(glass(h_c_bottom=20000.0, h_c_top=1000.0), numerics=FAST)
    assert abs(a.dT[-1] / b.dT[-1] - 1) > 1e-4      # film stack is not symmetric
