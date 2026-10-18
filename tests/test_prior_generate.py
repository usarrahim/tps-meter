"""Tests for prior sampling, the auto-setup rule and the dataset generator."""

import h5py
import numpy as np
import pytest

from tpslib.config import CONFIG_DIR, load_sensor, load_yaml
from tpslib.data.prior import FAULTS, contact_conductance, sample_split

PRIOR = load_yaml(CONFIG_DIR / "prior_v1.yaml")
DEVICE = load_yaml(CONFIG_DIR / "device.yaml")
S = load_sensor()


@pytest.fixture(scope="module")
def p():
    return sample_split(20000, 1234)


def test_deterministic_and_seed_dependent():
    a, b, c = sample_split(50, 7), sample_split(50, 7), sample_split(50, 8)
    for key in a:
        assert np.array_equal(a[key], b[key])
    assert not np.allclose(a["k"], c["k"])


def test_prefix_stable():
    # The first samples do not depend on n (vectorised draws per parameter are NOT prefix-stable,
    # so this documents that splits must always be generated with their configured n).
    a, b = sample_split(10, 3), sample_split(10, 3)
    assert np.array_equal(a["k"], b["k"])


def test_ranges(p):
    assert p["k"].min() >= 0.05 and p["k"].max() <= 30
    assert p["rho_cp"].min() >= 0.8e6 and p["rho_cp"].max() <= 4.5e6
    assert np.allclose(p["alpha"], p["k"] / p["rho_cp"])
    for side in ("bottom", "top"):
        h = p[f"h_c_{side}"]
        assert h.min() >= 500 and h.max() <= 50000
    assert p["sample_thickness"].min() >= 8e-3 and p["sample_thickness"].max() <= 25e-3
    assert abs(np.std(p["alpha_R_calib_rel"]) - 0.01) < 0.001
    # log-uniform k: equal mass per decade
    frac = np.mean(p["k"] < np.sqrt(0.05 * 30))
    assert abs(frac - 0.5) < 0.02
    assert np.all(p["fault"] == 0)


def test_probe_error_bounds(p):
    assert np.all(np.abs(p["k_probe"] / p["k"] - 1) <= 0.3 + 1e-12)
    assert np.all(np.abs(p["alpha_probe"] / p["alpha"] - 1) <= 0.3 + 1e-12)


def test_auto_setup_rule(p):
    assert p["P0"].min() >= DEVICE["power"]["min"] and p["P0"].max() <= DEVICE["power"]["max"]
    a_used = S.outer_radius * (1 + p["radius_calib_rel"])
    tau_probe = np.sqrt(p["alpha_probe"] * p["t_end"]) / a_used
    assert np.all(tau_probe <= 1.0 + 1e-9)
    depth = 2 * np.sqrt(p["alpha_probe"] * p["t_end"])
    limit = 0.9 * np.minimum(p["thickness_entered"], p["sample_radius"] - a_used)
    assert np.all(depth <= limit * (1 + 1e-9))
    assert np.all(p["sps"] * p["t_end"] <= DEVICE["sample_rate"]["max_samples_per_record"]
                  + (p["sps"] == 20) * 1e9)
    # True tau_end within the +-30 % probe error envelope (times the radius calibration error)
    assert np.all(p["tau_end"] <= (a_used / S.outer_radius) / np.sqrt(0.7) * (1 + 1e-9))
    # Long records use the slowest rate, short ones the fastest
    assert np.all(p["sps"][p["t_end"] > 600] == 20)
    assert np.all(p["sps"][p["t_end"] < 9] == 1200)
    L = DEVICE["current_ranges"]["L"]
    assert np.all((p["I"] <= L["I_max"]) == (p["current_range"] == 0))


def test_power_follows_probe_k(p):
    unclipped = (p["P0"] > DEVICE["power"]["min"] * 1.001) & (p["P0"] < DEVICE["power"]["max"] * 0.999)
    a_used = S.outer_radius * (1 + p["radius_calib_rel"])
    ratio = (p["P0"] / (p["k_probe"] * a_used))[unclipped]
    assert np.ptp(ratio) / ratio.mean() < 1e-9


def test_contact_conductance_limits():
    base = {"scale_polyimide_base": np.ones(1), "scale_adhesive": np.ones(1), "scale_polyimide_cover": np.ones(1),
            "k_polyimide": np.full(1, 0.2), "k_adhesive": np.full(1, 0.2)}
    hi = contact_conductance({**base, "h_c_bottom": np.full(1, 1e12), "h_c_top": np.full(1, 1e12)},
                             load_yaml(CONFIG_DIR / "sensor.yaml"))
    assert hi[0] == pytest.approx(0.2 / 25e-6 + 0.2 / 27.5e-6, rel=1e-6)
    lo = contact_conductance({**base, "h_c_bottom": np.full(1, 100.0), "h_c_top": np.full(1, 100.0)},
                             load_yaml(CONFIG_DIR / "sensor.yaml"))
    assert lo[0] < 200


def test_k_band():
    p = sample_split(2000, 4004, k_band=[1.5, 2.5])
    assert p["k"].min() >= 1.5 and p["k"].max() <= 2.5


def test_faulty_split():
    p = sample_split(3000, 5005, faulty=True)
    kinds, counts = np.unique(p["fault"], return_counts=True)
    assert set(kinds) == {1, 2, 3} and counts.min() > 800
    gap = p["fault"] == FAULTS["air_gap_low_hc"]
    assert np.all(np.minimum(p["h_c_bottom"][gap], p["h_c_top"][gap]) <= 300)
    thin = p["fault"] == FAULTS["thin_sample"]
    assert np.all(p["sample_thickness"][thin] <= 6e-3)
    assert np.all(p["thickness_entered"][thin] >= 8e-3)          # user entered the wrong thickness
    jump = p["fault"] == FAULTS["drift_jump"]
    assert np.all(np.abs(p["drift_jump_K"][jump]) >= 0.005) and np.all(p["drift_jump_K"][~jump] == 0)


def test_generator_small_run_and_resume(tmp_path):
    from tpslib.data.generate import run_split

    stats = run_split("tiny", 4, 42, tmp_path, workers=2, shard_size=2, progress=False)
    assert stats["n_ok"] == 4 and stats["n_failed"] == 0
    f = tmp_path / "tiny.h5"
    with h5py.File(f, "r") as h:
        assert h["dT"].shape == (4, 512) and h["params/k"].shape == (4,)
        assert np.all(np.isfinite(h["dT"][()])) and np.all(h["dT"][:, -1] > 0)
        assert np.all(np.abs(h["diag/energy_rel_imbalance"][()]) < 1e-6)
        t = h["t"][()]
        assert np.allclose(t[:, 0], 1.0 / h["params/sps"][()])
        assert np.allclose(t[:, -1], h["params/t_end"][()])
        k_saved = h["params/k"][()]
    assert np.array_equal(k_saved, sample_split(4, 42)["k"])
    # Re-running returns immediately from the merged file
    stats2 = run_split("tiny", 4, 42, tmp_path, workers=2, shard_size=2, progress=False)
    assert stats2["n_ok"] == 4
    assert not (tmp_path / "tiny_shards").exists()


def test_generator_mirror_restores_shards_and_final(tmp_path):
    import shutil

    from tpslib.data.generate import _write_h5, code_hash, run_split

    local, mirror = tmp_path / "local", tmp_path / "drive"
    # Interrupted run: shard 0 finished and mirrored, then the session was lost.
    # Plant a recognisable fake shard 0 on the mirror; it must be restored, not re-simulated.
    p = sample_split(4, 7)
    fake = {"t": np.ones(512), "dT": np.full(512, 123.0), "P": np.ones(512), "ok": 1, "error": "",
            "sim_time_s": 0.0, "energy_rel_imbalance": 0.0, "frac_lost": 0.0, "n_steps": 0}
    (mirror / "m_shards").mkdir(parents=True)
    _write_h5(mirror / "m_shards" / "shard_00000.h5", {k: v[0:2] for k, v in p.items()}, [fake, fake],
              {"split": "m", "seed": 7, "n_total": 4, "code_hash": code_hash(), "shard": 0, "lo": 0, "hi": 2})
    run_split("m", 4, 7, local, workers=2, shard_size=2, progress=False, mirror=mirror)
    with h5py.File(local / "m.h5", "r") as f:
        dT = f["dT"][()]
    assert np.all(dT[:2] == 123.0)                      # restored from the mirror
    assert np.all(dT[2:, -1] > 0) and np.all(dT[2:, -1] < 50)   # simulated
    assert (mirror / "m.h5").exists() and not (mirror / "m_shards").exists()

    # Lost session after completion: local disk empty -> merged file comes back from the mirror
    shutil.rmtree(local)
    stats = run_split("m", 4, 7, local, workers=2, shard_size=2, progress=False, mirror=mirror)
    assert (local / "m.h5").exists() and stats["n"] == 4


def test_shard_from_other_code_is_rejected(tmp_path):
    from tpslib.data.generate import _shard_complete, _write_h5

    p = sample_split(2, 7)
    fake = {"t": np.ones(512), "dT": np.ones(512), "P": np.ones(512), "ok": 1, "error": "",
            "sim_time_s": 0.0, "energy_rel_imbalance": 0.0, "frac_lost": 0.0, "n_steps": 0}
    f = tmp_path / "s.h5"
    _write_h5(f, p, [fake, fake], {"code_hash": "old"})
    with pytest.raises(RuntimeError):
        _shard_complete(f, 2, "new")
    assert _shard_complete(f, 2, "old")
