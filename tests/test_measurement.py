"""Phase 4 tests: measurement chain, shared preprocessing, features, dataset and normaliser."""

import copy

import h5py
import numpy as np
import pytest

from tpslib.config import CONFIG_DIR, load_sensor, load_yaml
from tpslib.data.features import CHANNELS, SCALARS, make_features, time_grid
from tpslib.dtau import PI15, dtau, ideal_rise_constant_current
from tpslib.fit_tps import fit_tps
from tpslib.measurement.acquisition import acquire
from tpslib.measurement.preprocess import preprocess
from tpslib.synth_analytic import clean_record_case

S = load_sensor()
A, M = S.outer_radius, S.n_rings
DEVICE = load_yaml(CONFIG_DIR / "device.yaml")


# Kept under this name for the other test modules that import it.
analytic_case = clean_record_case


def test_noiseless_chain_is_exact_inverse():
    t, dT, prm = analytic_case(1.2)
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects=False))
    ref = np.interp(np.log(p.t), np.log(t), dT)
    assert np.max(np.abs(p.dT - ref)) < 1e-6
    assert np.allclose(p.P, prm["P0"] * (1 + S.alpha_R * p.dT), rtol=1e-9)
    assert p.t[0] == pytest.approx(1 / prm["sps"])
    assert prm["t_end"] - 1 / prm["sps"] < p.t[-1] <= prm["t_end"]


def test_alpha_R_and_radius_calibration_errors():
    dev = copy.deepcopy(DEVICE)
    dev["calibration"]["R0_rel_std"] = 0.0
    t, dT, prm = analytic_case(1.2, alpha_R_calib_rel=0.01, radius_calib_rel=0.005)
    rec = acquire(t, dT, prm, np.random.default_rng(0), device=dev, effects={"calibration": True})
    p = preprocess(rec)
    p0 = preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects=False))
    # analysis divides by the (wrong) calibrated alpha_R -> dT scaled by 1/(1+err); R0/R_ref errors are tiny
    assert np.median(p.dT[-100:] / p0.dT[-100:]) == pytest.approx(1 / 1.01, rel=2e-4)
    assert p.a == pytest.approx(A * 1.005)


def test_R_ref_tolerance_cancels():
    dev = copy.deepcopy(DEVICE)
    dev["calibration"].update(R0_rel_std=0.0, R_ref_tolerance=0.01)
    t, dT, prm = analytic_case(1.2)
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(3), device=dev, effects={"calibration": True}))
    p0 = preprocess(acquire(t, dT, prm, np.random.default_rng(3), effects=False))
    assert np.max(np.abs(p.dT - p0.dT)) < 1e-6


def test_drift_is_removed_up_to_estimate_error_and_quadratic_part():
    t, dT, prm = analytic_case(0.3, rho_cp=1.5e6)
    rec = acquire(t, dT, prm, np.random.default_rng(5), effects={"drift": True})
    p = preprocess(rec)
    p0 = preprocess(acquire(t, dT, prm, np.random.default_rng(5), effects=False))
    tr = rec.truth
    expected = (tr["drift_d1"] - tr["drift_slope_est"]) * p.t + tr["drift_d2"] * p.t ** 2
    assert np.max(np.abs((p.dT - p0.dT) - expected)) < 1e-6


def test_white_noise_level():
    dev = copy.deepcopy(DEVICE)
    dev["noise"].update(flicker_corner_hz=[0.0, 0.0], noise_scale=[1.0, 1.0])
    t, dT, prm = analytic_case(1.2)
    rec = acquire(t, dT, prm, np.random.default_rng(7), device=dev, effects={"noise": True})
    p = preprocess(rec)
    p0 = preprocess(acquire(t, dT, prm, np.random.default_rng(7), effects=False))
    sigma = np.std(p.dT - p0.dT)
    assert sigma == pytest.approx(rec.truth["sigma_T"], rel=0.05)
    # datasheet: 0.23 uV at 20 SPS ~ 0.15 mK at 100 mA; scales with 1/I and sqrt(rate)
    I = np.sqrt(prm["P0"] / S.R0)
    expect = 0.23e-6 * np.sqrt(prm["sps"] / 20) / (I * S.R0 * S.alpha_R)
    assert rec.truth["sigma_T"] == pytest.approx(expect, rel=1e-9)


def test_drift_jump_fault():
    t, dT, prm = analytic_case(1.2, drift_jump_K=0.03, drift_jump_t_frac=0.5)
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects={"fault": True}))
    p0 = preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects=False))
    d = p.dT - p0.dT
    half = p.t >= 0.5 * prm["t_end"]
    assert np.allclose(d[half], 0.03, atol=1e-6) and np.allclose(d[~half], 0.0, atol=1e-6)


# --- features ---------------------------------------------------------------------------------

def test_features_shapes_and_ideal_theta():
    t, dT, prm = analytic_case(1.2)
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(0), effects=False))
    F = make_features(p)
    assert F["seq"].shape == (len(CHANNELS), 256) and F["scalars"].shape == (len(SCALARS),)
    assert np.all(np.isfinite(F["seq"])) and np.all(np.isfinite(F["scalars"]))
    # ideal curve: k theta = D(tau) + const (power feedback adds only ~alpha_R dT ~ 1 %)
    tau = np.sqrt(prm["alpha"] * F["t_grid"]) / A
    kth = prm["k"] * F["theta"]
    late = tau > 0.3
    off = kth[late] - dtau(tau[late])
    assert np.ptp(off) < 0.01 * kth[late].max()
    # derivative channel vs analytic d(D)/d ln t
    dD = np.gradient(dtau(tau), np.log(F["t_grid"]))
    theta_ref = np.exp(F["scalars"][0])
    num = F["seq"][1] * theta_ref * prm["k"]
    assert np.max(np.abs(num[late] - dD[late])) < 0.02 * dD[late].max()


def test_grid_covers_three_decades():
    edges, centers = time_grid(100.0)
    assert edges[0] == pytest.approx(0.1) and edges[-1] == pytest.approx(100.0)
    assert centers.size == 256


# --- Phase 4 gate: M0 still works on noisy ideal curves -----------------------------------------

K_VALUES = np.geomspace(0.05, 30.0, 20)


@pytest.mark.parametrize("k", K_VALUES)
def test_gate_m0_on_noisy_ideal_curves(k):
    rho_cp = np.interp(np.log(k), np.log([0.05, 30]), [1.0e6, 4.0e6])
    t, dT, prm = analytic_case(float(k), rho_cp=rho_cp)
    eff = {"noise": True, "drift": True, "timing": True, "power": True, "electronic": True}
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(int(k * 1000)), effects=eff))
    r = fit_tps(p.t, p.dT, p.P, p.a, M)
    assert abs(r.k / prm["k"] - 1) < 0.02
    # alpha is far more sensitive: the uncorrectable quadratic part of the ambient drift moves the
    # M0 alpha by up to ~10 % on long low-k records (reports/phase4.md); only a sanity bound here.
    assert abs(r.alpha / prm["alpha"] - 1) < 0.2


@pytest.mark.parametrize("k", [0.05, 1.2, 16.0])
def test_m0_alpha_with_noise_only(k):
    t, dT, prm = analytic_case(k, rho_cp=2.0e6)
    p = preprocess(acquire(t, dT, prm, np.random.default_rng(11), effects={"noise": True, "timing": True}))
    r = fit_tps(p.t, p.dT, p.P, p.a, M)
    assert abs(r.k / prm["k"] - 1) < 0.01 and abs(r.alpha / prm["alpha"] - 1) < 0.03


# --- dataset / normaliser -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_h5(tmp_path_factory):
    path = tmp_path_factory.mktemp("ds") / "tiny.h5"
    cases = [analytic_case(k, n_grid=512) for k in (0.1, 0.5, 2.0, 2.2, 8.0, 20.0)]
    with h5py.File(path, "w") as f:
        g = f.create_group("params")
        for key in cases[0][2]:
            g.create_dataset(key, data=np.array([c[2][key] for c in cases], dtype=float))
        f.create_dataset("t", data=np.stack([c[0] for c in cases]))
        f.create_dataset("dT", data=np.stack([c[1] for c in cases]).astype(np.float32))
        f.create_group("diag").create_dataset("ok", data=np.ones(len(cases), int))
    return path


def test_dataset_eval_fixed_train_varies(tiny_h5):
    torch = pytest.importorskip("torch")
    from tpslib.data.dataset import TPSDataset

    ev = TPSDataset(tiny_h5, train=False, base_seed=2002)
    a = ev[0][0].numpy(); ev.set_epoch(5); b = ev[0][0].numpy()
    assert np.array_equal(a, b)
    tr = TPSDataset(tiny_h5, train=True, base_seed=1001)
    a = tr[0][0].numpy(); tr.set_epoch(1); b = tr[0][0].numpy(); tr.set_epoch(0); c = tr[0][0].numpy()
    assert not np.array_equal(a, b) and np.array_equal(a, c)
    seq, scal, y = tr[1]
    assert seq.dtype == torch.float32 and y.shape == (3,)
    assert y[0].item() == pytest.approx(np.log(0.5), rel=1e-6)


def test_normalizer_train_only_and_roundtrip(tiny_h5, tmp_path):
    from tpslib.data.dataset import Normalizer, TPSDataset, select_indices

    with pytest.raises(ValueError):
        Normalizer.fit(TPSDataset(tiny_h5, train=False, base_seed=1))
    ds = TPSDataset(tiny_h5, train=True, base_seed=1)
    nm = Normalizer.fit(ds, n=6)
    nm.save(tmp_path / "norm.json")
    nm2 = Normalizer.load(tmp_path / "norm.json")
    assert np.allclose(nm.seq_mean, nm2.seq_mean) and nm2.fitted_on == "tiny.h5"
    ds.normalizer = nm2
    seq, scal, y = ds[0]
    assert np.all(np.isfinite(seq.numpy()))
    # ablation: exclude the interpolation band 1.5-2.5 W/m/K
    idx = select_indices(ds.params, exclude_k_band=(1.5, 2.5))
    assert set(np.round(ds.params["k"][idx], 3)) == {0.1, 0.5, 8.0, 20.0}
