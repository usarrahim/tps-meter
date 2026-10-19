"""Clean simulated curve -> raw device record (what the board would log), with measurement effects.

The output `RawRecord` has the same fields a real measurement provides (plan 7.3, 8.3, 9.2):
  ADC1 ratio R_sensor/R_ref at the sample times, ADC2 absolute current, R0 and the drift slope from the
  pre-measurement, and the sensor calibration (alpha_R, radius) the analysis will use.
Real and synthetic records then go through the same `preprocess` (PROJECT_RULES.md data rules).

Effects (device.yaml, all random per call):
  - sampling at the device rate, clock error, heating-start offset t_c
  - ambient drift: linear + small quadratic; the pre-measurement slope estimate has an error
  - ADC white + 1/f noise (voltage noise / (I R0 alpha_R)), 32-bit quantisation of the ratio
  - power error (absolute current), ADC2 current noise
  - R0 measurement error, R_ref tolerance (cancels ratiometrically)
  - electronic drift (R_ref self-heating) with an imperfect correction curve
  - calibration errors of alpha_R and of the sensor radius (prior parameters of the sample)
  - fault "drift_jump" (test_faulty): a step in the sensor temperature
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tpslib.config import CONFIG_DIR, load_yaml


@dataclass
class RawRecord:
    """One measurement in the device's raw format (shared by real and synthetic data)."""
    t: np.ndarray            # sample times after the heating command [s] (device clock)
    ratio: np.ndarray        # ADC1: R_sensor / R_ref
    R_ref: float             # nominal reference resistor of the active range [ohm]
    t_I: np.ndarray          # ADC2 sample times [s]
    I: np.ndarray            # ADC2: measured current [A]
    R0: float                # sensor resistance from the pre-measurement [ohm]
    drift_slope: float       # ambient drift from the pre-measurement [K/s]
    alpha_R: float           # calibrated TCR used by the analysis [1/K]
    a: float                 # calibrated sensor radius used by the analysis [m]
    sps: float
    current_range: int       # 0 = L, 1 = H
    thickness: float         # entered by the user [m]
    radius: float            # entered by the user [m]
    elec_amp_K_per_W: float  # electronic-drift correction curve (dummy-load test)
    elec_tau_s: float
    truth: dict = field(default_factory=dict)  # synthetic only: the effects actually applied


def _interp_clean(t_query, t_clean, y_clean):
    """Clean curve at arbitrary times: linear in ln t inside the grid, ~sqrt(t) before it, 0 for t <= 0."""
    out = np.zeros_like(t_query)
    pos = t_query > 0
    tq = t_query[pos]
    inside = np.interp(np.log(np.maximum(tq, t_clean[0])), np.log(t_clean), y_clean)
    early = tq < t_clean[0]
    inside[early] = y_clean[0] * np.sqrt(tq[early] / t_clean[0])
    out[pos] = inside
    return out


def colored_noise(rng, n: int, sps: float, sigma: float, f_corner: float) -> np.ndarray:
    """White noise of rms `sigma` (per sample) plus 1/f noise with PSD = white * f_corner / f."""
    white = rng.standard_normal(n)
    if f_corner <= 0 or n < 4:
        return sigma * white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n, 1.0 / sps)
    f[0] = f[1]
    spec *= np.sqrt(1.0 + f_corner / f)
    return sigma * np.fft.irfft(spec, n)


def acquire(t_clean, dT_clean, prm: dict, rng: np.random.Generator, device: dict | None = None,
            sensor: dict | None = None, effects: bool | dict = True) -> RawRecord:
    """Make a raw device record from a clean curve.

    t_clean, dT_clean : clean mean sensor temperature rise on the simulator's log grid
    prm               : sample parameters (prior row: P0, sps, t_end, calibration errors, fault, ...)
    effects           : True (all), False (none: exact inverse of preprocess), or dict of switches
                        {noise, drift, timing, calibration, power, electronic, fault}
    """
    device = device or load_yaml(CONFIG_DIR / "device.yaml")
    sensor = sensor or load_yaml(CONFIG_DIR / "sensor.yaml")
    names = ("noise", "drift", "timing", "calibration", "power", "electronic", "fault")
    if isinstance(effects, dict):
        unknown = set(effects) - set(names)
        if unknown:
            raise ValueError(f"unknown effects {unknown}")
        on = {k: bool(effects.get(k, False)) for k in names}
    else:
        on = {k: bool(effects) for k in names}
    nz, dr, tm, cal, el = (device["noise"], device["drift"], device["timing"], device["calibration"],
                           device["electronic_drift"])

    R0_true = float(sensor["electrical"]["R0"])
    aR_true = float(sensor["electrical"]["alpha_R"])
    a_true = float(sensor["spiral"]["outer_radius"])
    P0, sps, t_end = float(prm["P0"]), float(prm["sps"]), float(prm["t_end"])
    rng_name = "L" if int(prm["current_range"]) == 0 else "H"
    R_ref_nom = float(device["current_ranges"][rng_name]["R_ref"])

    # --- sampling ---------------------------------------------------------------------------
    n = max(int(np.floor(t_end * sps)), 8)
    t = np.arange(1, n + 1) / sps
    clock = 1.0 + (rng.normal(0.0, tm["clock_ppm_std"] * 1e-6) if on["timing"] else 0.0)
    t_c = rng.uniform(-tm["t_c_max"], tm["t_c_max"]) if on["timing"] else 0.0
    t_heat = t * clock - t_c                        # true time since the heating started
    dT_true = _interp_clean(t_heat, np.asarray(t_clean, float), np.asarray(dT_clean, float))

    # --- ambient drift ----------------------------------------------------------------------
    if on["drift"]:
        d1 = rng.uniform(-dr["linear_max"], dr["linear_max"])
        d2 = dr["quadratic_frac"] * abs(d1) / t_end * rng.uniform(-1.0, 1.0)
        slope_est = d1 + rng.normal(0.0, dr["slope_estimate_std"])
    else:
        d1 = d2 = slope_est = 0.0
    drift = d1 * t + d2 * t * t
    jump = 0.0
    if on["fault"] and float(prm.get("drift_jump_K", 0.0)) != 0.0:
        jump = float(prm["drift_jump_K"])
        drift = drift + jump * (t >= float(prm["drift_jump_t_frac"]) * t_end)

    # --- electronic drift (apparent, from R_ref self-heating) -------------------------------
    elec_amp, elec_tau = float(el["amp_K_per_W"]), float(el["tau_s"])
    elec_true = 0.0
    if on["electronic"]:
        u = rng.uniform(-el["residual_frac"], el["residual_frac"])
        elec_true = elec_amp * (1.0 + u) * P0 * (1.0 - np.exp(-t / elec_tau))

    # --- sensor resistance and ratio --------------------------------------------------------
    T_sensor = dT_true + drift
    R_true = R0_true * (1.0 + aR_true * (T_sensor + elec_true))
    I_true = np.sqrt(P0 / R0_true)
    if on["noise"]:
        scale = rng.uniform(*nz["noise_scale"])
        fc = rng.uniform(*nz["flicker_corner_hz"])
        v_n = nz["white_uV_at_20sps"] * 1e-6 * (sps / 20.0) ** nz["white_rate_exponent"] * scale
        sigma_T = v_n / (I_true * R0_true * aR_true)
        R_true = R_true + R0_true * aR_true * colored_noise(rng, n, sps, sigma_T, fc)
    else:
        scale = fc = sigma_T = 0.0
    R_ref_true = R_ref_nom * (1.0 + (rng.uniform(-cal["R_ref_tolerance"], cal["R_ref_tolerance"])
                                     if on["calibration"] else 0.0))
    ratio = R_true / R_ref_true
    lsb = 2.0 ** -(int(nz["adc_bits"]) - 1)
    ratio = np.round(ratio / lsb) * lsb

    # --- pre-measurement and ADC2 -----------------------------------------------------------
    e_R0 = rng.normal(0.0, cal["R0_rel_std"]) if on["calibration"] else 0.0
    R0_meas = R0_true * (R_ref_nom / R_ref_true) * (1.0 + e_R0)
    e_P = rng.uniform(-cal["power_error_rel"], cal["power_error_rel"]) if on["power"] else 0.0
    t_I = np.arange(1, max(int(np.floor(t_end * nz["adc2_rate"])), 1) + 1) / nz["adc2_rate"]
    I_meas = I_true * np.sqrt(1.0 + e_P) * np.ones_like(t_I)
    if on["noise"]:
        I_meas = I_meas * (1.0 + rng.normal(0.0, nz["adc2_current_noise_rel"], t_I.size))

    # --- calibration the analysis believes in ----------------------------------------------
    aR_cal = aR_true * (1.0 + (float(prm.get("alpha_R_calib_rel", 0.0)) if on["calibration"] else 0.0))
    a_cal = a_true * (1.0 + (float(prm.get("radius_calib_rel", 0.0)) if on["calibration"] else 0.0))

    truth = {"clock": clock, "t_c": t_c, "drift_d1": d1, "drift_d2": d2, "drift_slope_est": slope_est,
             "jump_K": jump, "noise_scale": scale, "flicker_corner_hz": fc, "sigma_T": sigma_T,
             "power_error": e_P, "R0_error": e_R0, "R_ref_true": R_ref_true}
    return RawRecord(t=t, ratio=ratio, R_ref=R_ref_nom, t_I=t_I, I=I_meas, R0=R0_meas,
                     drift_slope=slope_est, alpha_R=aR_cal, a=a_cal, sps=sps,
                     current_range=int(prm["current_range"]),
                     thickness=float(prm.get("thickness_entered", prm.get("sample_thickness", np.nan))),
                     radius=float(prm.get("sample_radius", np.nan)),
                     elec_amp_K_per_W=elec_amp if on["electronic"] else 0.0, elec_tau_s=elec_tau, truth=truth)
