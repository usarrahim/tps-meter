"""Parameter sampling (prior_v1.yaml) and the device auto-setup rule.

All draws for a split are made up front, vectorised, in a fixed order from one seeded generator,
so the parameters do not depend on the number of worker processes.

The auto-setup rule mirrors the device firmware (plan 10.3): from a probe estimate of k and alpha
(each off by up to +-30 %) it picks
  power     : ideal TPS rise of `target_dT` at tau = 1, clipped to the device power range,
  duration  : min(tau_max^2 a^2 / alpha_probe, time for which 2 sqrt(alpha_probe t) = 0.9 min(H, R - a)),
  rate      : fastest allowed sample rate within the per-record sample budget,
  range     : L if the current fits in it, else H.
Power and duration are therefore NOT sampled independently.
"""

from __future__ import annotations

import numpy as np

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.dtau import PI15, dtau

FILM_LAYERS = ("polyimide_base", "copper_level", "adhesive", "polyimide_cover")
FAULTS = {"none": 0, "air_gap_low_hc": 1, "thin_sample": 2, "drift_jump": 3}


def _draw(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    d = spec["dist"]
    if d == "uniform":
        return rng.uniform(spec["low"], spec["high"], n)
    if d == "loguniform":
        return np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]), n))
    if d == "normal":
        return rng.normal(spec["mean"], spec["std"], n)
    raise ValueError(f"unknown distribution {d!r}")


def contact_conductance(p: dict, sensor_cfg: dict) -> np.ndarray:
    """Total thermal conductance per area from the copper level to both samples [W/m^2/K].

    G_c = 1/(R_below + 1/h_bottom) + 1/(R_above + 1/h_top), R = film layer resistances between the
    copper and each sample. This is the identifiable "contact" quantity the model predicts
    (DECISIONS.md, Phase 0 question 1).
    """
    t = {s["name"]: float(s["thickness"]) for s in sensor_cfg["film"]["stack"]}
    R_below = t["polyimide_base"] * p["scale_polyimide_base"] / p["k_polyimide"]
    R_above = (t["adhesive"] * p["scale_adhesive"] / p["k_adhesive"]
               + t["polyimide_cover"] * p["scale_polyimide_cover"] / p["k_polyimide"])
    return 1.0 / (R_below + 1.0 / p["h_c_bottom"]) + 1.0 / (R_above + 1.0 / p["h_c_top"])


def auto_setup(k_probe, alpha_probe, thickness_entered, radius_entered, a, m, R0, device, rule):
    """Vectorised firmware auto-setup. Returns dict of P0, I, current_range, t_end, sps."""
    k_probe, alpha_probe = np.asarray(k_probe, float), np.asarray(alpha_probe, float)
    D1 = float(dtau(np.array(rule["tau_max"]), m))
    P = rule["target_dT"] * PI15 * a * k_probe / D1
    P = np.clip(P, device["power"]["min"], device["power"]["max"])
    t_tau = rule["tau_max"] ** 2 * a * a / alpha_probe
    depth = rule["max_depth_frac"] * np.minimum(thickness_entered, radius_entered - a)
    t_depth = (depth / 2.0) ** 2 / alpha_probe
    t_end = np.minimum(t_tau, t_depth)
    cap = device["sample_rate"]["max_samples_per_record"]
    rates = np.array(sorted(device["sample_rate"]["allowed"]), float)
    fits = rates[None, :] * t_end[:, None] <= cap
    sps = np.where(fits.any(axis=1), np.max(np.where(fits, rates[None, :], 0.0), axis=1), rates[0])
    I = np.sqrt(P / R0)
    L = device["current_ranges"]["L"]
    rng_idx = np.where(I <= L["I_max"], 0, 1)   # 0 = L, 1 = H
    return {"P0": P, "I": I, "current_range": rng_idx, "t_end": t_end, "sps": sps,
            "t_end_limited_by_depth": (t_depth < t_tau).astype(np.int8)}


def sample_split(n: int, seed: int, prior: dict | None = None, sensor_cfg: dict | None = None,
                 device: dict | None = None, k_band=None, faulty: bool = False) -> dict[str, np.ndarray]:
    """Sample n parameter sets for one split. Returns a dict of equal-length arrays."""
    prior = prior or load_yaml(CONFIG_DIR / "prior_v1.yaml")
    sensor_cfg = sensor_cfg or load_yaml(CONFIG_DIR / "sensor.yaml")
    device = device or load_yaml(CONFIG_DIR / "device.yaml")
    rng = np.random.default_rng(seed)
    a = float(sensor_cfg["spiral"]["outer_radius"])
    m = int(sensor_cfg["spiral"]["n_rings"])
    R0 = float(sensor_cfg["electrical"]["R0"])

    p: dict[str, np.ndarray] = {}
    if k_band is None:
        p["k"] = _draw(rng, prior["k"], n)
    else:
        p["k"] = _draw(rng, {"dist": "loguniform", "low": k_band[0], "high": k_band[1]}, n)
    p["rho_cp"] = _draw(rng, prior["rho_cp"], n)
    p["alpha"] = p["k"] / p["rho_cp"]
    p["h_c_bottom"] = _draw(rng, prior["h_c_bottom"], n)
    p["h_c_top"] = _draw(rng, prior["h_c_top"], n)
    for name in FILM_LAYERS:
        p[f"scale_{name}"] = _draw(rng, prior["layer_thickness_rel"], n)
    p["k_polyimide"] = _draw(rng, prior["k_polyimide"], n)
    p["k_adhesive"] = _draw(rng, prior["k_adhesive"], n)
    p["sample_radius"] = _draw(rng, prior["sample_radius"], n)
    p["sample_thickness"] = _draw(rng, prior["sample_thickness"], n)
    p["alpha_R_calib_rel"] = _draw(rng, prior["alpha_R_calib_rel"], n)
    p["radius_calib_rel"] = _draw(rng, prior["radius_calib_rel"], n)
    e = prior["auto_setup"]["probe_error_rel"]
    p["k_probe"] = p["k"] * (1.0 + rng.uniform(-e, e, n))
    p["alpha_probe"] = p["alpha"] * (1.0 + rng.uniform(-e, e, n))

    # Faults (test_faulty): drawn after the regular parameters so the regular draws are unchanged
    p["fault"] = np.zeros(n, np.int8)
    p["thickness_entered"] = p["sample_thickness"].copy()
    p["drift_jump_K"] = np.zeros(n)
    p["drift_jump_t_frac"] = np.zeros(n)
    if faulty:
        fc = prior["faults"]
        names = list(fc)
        w = np.array([fc[f]["weight"] for f in names], float)
        kind = rng.choice(len(names), size=n, p=w / w.sum())
        for j, name in enumerate(names):
            sel = kind == j
            ns = int(sel.sum())
            p["fault"][sel] = FAULTS[name]
            if name == "air_gap_low_hc":
                hc = _draw(rng, fc[name]["h_c"], ns)
                side = rng.integers(0, len(fc[name]["sides"]), ns)
                sides = np.array(fc[name]["sides"])[side]
                idx = np.where(sel)[0]
                bot = np.isin(sides, ["bottom", "both"])
                top = np.isin(sides, ["top", "both"])
                p["h_c_bottom"][idx[bot]] = hc[bot]
                p["h_c_top"][idx[top]] = hc[top]
            elif name == "thin_sample":
                p["sample_thickness"][sel] = _draw(rng, fc[name]["true_thickness"], ns)
            elif name == "drift_jump":
                p["drift_jump_K"][sel] = _draw(rng, fc[name]["jump_K"], ns) * rng.choice([-1.0, 1.0], ns)
                p["drift_jump_t_frac"][sel] = rng.uniform(0.2, 0.9, ns)

    a_used = a * (1.0 + p["radius_calib_rel"])
    setup = auto_setup(p["k_probe"], p["alpha_probe"], p["thickness_entered"], p["sample_radius"],
                       a_used, m, R0, device, prior["auto_setup"])
    p.update(setup)
    p["tau_end"] = np.sqrt(p["alpha"] * p["t_end"]) / a
    p["G_c"] = contact_conductance(p, sensor_cfg)
    p["seed"] = np.full(n, seed, np.int64)
    p["index"] = np.arange(n, dtype=np.int64)
    return p


def row(p: dict, i: int) -> dict:
    """Parameters of sample i as plain Python scalars."""
    return {k: v[i].item() for k, v in p.items()}
