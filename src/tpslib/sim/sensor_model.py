"""High-level simulator: sensor + two samples, constant-current feedback, length-weighted readout.

Ring sources: Joule heat uniform per copper volume, i.e. ring power proportional to ring length
(every ring has the same cross-section b x t_Cu and carries the same current).
Readout: length-weighted mean of the ring temperatures (exactly what the 4-wire resistance measures).
Constant current: P(t) = I^2 R0 (1 + alpha_R dT_mean(t)), implicit in the solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.sim.geometry import build_mesh
from tpslib.sim.solver import assemble, integrate, integrate_trbdf2


@dataclass
class SimParams:
    k: float                          # sample conductivity [W/m/K]
    rho_cp: float                     # sample volumetric heat capacity [J/m^3/K]
    P0: float                         # power at t = 0+ (I^2 R0) [W]
    t_end: float                      # [s]
    h_c_bottom: float = np.inf        # contact conductance sample/film, bottom [W/m^2/K]
    h_c_top: float = np.inf
    sample_radius: float = 25e-3      # [m]
    thickness_bottom: float = 20e-3   # [m]
    thickness_top: float = 20e-3      # [m]
    k_polyimide: float | None = None  # overrides of film materials [W/m/K]
    k_adhesive: float | None = None
    layer_scale: dict = field(default_factory=dict)  # thickness multipliers per layer name
    constant_current: bool = True
    alpha_R: float | None = None      # true TCR used in the feedback (default: sensor config)
    layers: bool = True               # False: analytic-limit configuration (no film)
    h_convection: float | None = None # default from sim_numerics.yaml
    t_out_min: float | None = None    # first stored time [s] (e.g. 1/sample rate); default from config

    @property
    def alpha(self) -> float:
        return self.k / self.rho_cp


@dataclass
class SimResult:
    t: np.ndarray                     # log-time output grid [s]
    dT: np.ndarray                    # mean sensor temperature rise [K]
    P: np.ndarray                     # power [W]
    t_steps: np.ndarray               # raw integrator times
    dT_steps: np.ndarray
    energy: dict
    info: dict


def load_numerics() -> dict:
    return load_yaml(CONFIG_DIR / "sim_numerics.yaml")


def load_sensor_raw() -> dict:
    return load_yaml(CONFIG_DIR / "sensor.yaml")


def simulate(p: SimParams, sensor_cfg: dict | None = None, numerics: dict | None = None,
             refine: int = 1, t_out: np.ndarray | None = None) -> SimResult:
    t0 = time.perf_counter()
    sc = sensor_cfg or load_sensor_raw()
    nu = numerics or load_numerics()
    mats = sc["materials"]
    override = {}
    if p.k_polyimide is not None:
        override["polyimide"] = {"k": p.k_polyimide}
    if p.k_adhesive is not None:
        override["adhesive"] = {"k": p.k_adhesive}
    mesh = build_mesh(sc, nu, p.sample_radius, p.thickness_bottom, p.thickness_top,
                      materials_override=override, layer_scale=p.layer_scale,
                      layers_on=p.layers, refine=refine)
    h_conv = float(nu["boundary"]["h_convection"]) if p.h_convection is None else p.h_convection
    sys = assemble(mesh, p.k, p.rho_cp, p.k, p.rho_cp, p.h_c_bottom, p.h_c_top,
                   h_conv, float(mats["air"]["k"]))
    t_asm = time.perf_counter() - t0
    alpha_R = float(sc["electrical"]["alpha_R"]) if p.alpha_R is None else p.alpha_R
    c_fb = alpha_R if p.constant_current else 0.0
    tn = nu["time"]
    if t_out is None:
        t_min = p.t_out_min if p.t_out_min is not None else float(nu["output"]["t_min"])
        t_out = np.geomspace(t_min, p.t_end, int(nu["output"]["n_points"]))
    # Slow records (low sample rate) need no resolution far below their first output time.
    dt0 = max(float(tn["dt0"]), float(tn["dt0_fraction"]) * float(t_out[0])) / refine
    scheme = {"trbdf2": integrate_trbdf2, "richardson_ie": integrate}[tn.get("scheme", "richardson_ie")]
    res = scheme(sys, p.P0, c_fb, p.t_end, dt0, int(tn["steps_per_level"]) * refine)

    spline = CubicSpline(np.log(res.t), res.Tbar)
    dT = spline(np.log(t_out))
    P = p.P0 * (1.0 + c_fb * dT)

    stored = float(sys.C @ res.T_final)
    stored_samples = float(sys.C[sys.sample_mask] @ res.T_final[sys.sample_mask])
    energy = {
        "E_in": res.E_in,
        "E_stored": stored,
        "E_loss": res.E_loss,
        "rel_imbalance": (res.E_in - stored - res.E_loss) / res.E_in,
        "frac_stored_samples": stored_samples / res.E_in,
        "frac_stored_film": (stored - stored_samples) / res.E_in,
        "frac_lost": res.E_loss / res.E_in,
        "t_final": float(res.t[-1]),
    }
    info = {
        "n_unknowns": sys.n,
        "n_r": mesh.nr,
        "n_r_coarse": mesh.r_faces_coarse.size - 1,
        "n_z": mesh.bottom.dz.size + mesh.top.dz.size,
        "n_z_near": mesh.bottom.n_near + mesh.top.n_near,
        "n_steps": res.t.size,
        "n_factorizations": res.n_factorizations,
        "time_assemble_s": t_asm,
        "time_total_s": time.perf_counter() - t0,
    }
    return SimResult(t_out, dT, P, res.t, res.Tbar, energy, info)
