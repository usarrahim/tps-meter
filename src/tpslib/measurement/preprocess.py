"""Shared preprocessing: raw device record -> temperature rise dT(t) and power P(t) (plan 9.2).

Real device data and synthetic records (measurement.acquisition) both go through `preprocess`.

  1. R(t) = ratio * R_ref                       (ratiometric: R_ref errors cancel in R/R0)
  2. dT(t) = (R(t)/R0 - 1) / alpha_R            (R0 and alpha_R from pre-measurement / calibration)
  3. subtract the linear drift measured in the pre-measurement
  4. subtract the electronic-drift curve from the dummy-load test
  5. P(t) = I(t)^2 R(t), I from ADC2 (calibrated)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tpslib.measurement.acquisition import RawRecord


@dataclass
class Processed:
    t: np.ndarray        # [s]
    dT: np.ndarray       # mean sensor temperature rise [K]
    P: np.ndarray        # heating power [W]
    a: float             # sensor radius used by the analysis [m]
    sps: float
    current_range: int
    thickness: float     # [m]
    radius: float        # [m]

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def P0(self) -> float:
        return float(self.P[0])


def preprocess(rec: RawRecord) -> Processed:
    R = rec.ratio * rec.R_ref
    dT = (R / rec.R0 - 1.0) / rec.alpha_R
    dT = dT - rec.drift_slope * rec.t
    I = np.interp(rec.t, rec.t_I, rec.I)
    if rec.elec_amp_K_per_W:
        P_est = float(np.mean(I) ** 2 * rec.R0)
        dT = dT - rec.elec_amp_K_per_W * P_est * (1.0 - np.exp(-rec.t / rec.elec_tau_s))
    P = I * I * R
    return Processed(t=rec.t, dT=dT, P=P, a=rec.a, sps=rec.sps, current_range=rec.current_range,
                     thickness=rec.thickness, radius=rec.radius)
