"""Model inputs from a preprocessed record (real or synthetic): log-time grid, channels, scalars.

Channels on a 256-point log-time grid covering [t_end * span_ratio, t_end]:
  theta_rel        theta / theta_ref,           theta = dT pi^1.5 a / P  (ideally D(tau)/k + const)
  dtheta_dlnt_rel  d theta / d ln t / theta_ref (Savitzky-Golay, local quadratic in ln t)
  log_count        ln(1 + samples in the bin)   (tells the model how noisy each bin is)
theta spans ~600x over the k range, so its level is carried separately by the scalar ln(theta_ref)
(theta_ref = mean theta over the last bins); no information is lost by the scaling.

Scalars: ln theta_ref, ln t_end, ln P0, dT_max, entered thickness [mm], entered radius [mm], current range.
Targets: ln k, ln alpha, ln G_c.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from tpslib.config import CONFIG_DIR, load_yaml
from tpslib.dtau import PI15
from tpslib.measurement.preprocess import Processed

_CFG = load_yaml(CONFIG_DIR / "features.yaml")
CHANNELS = list(_CFG["channels"])
SCALARS = list(_CFG["scalars"])
TARGETS = list(_CFG["targets"])


def time_grid(t_end: float, cfg: dict = _CFG) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and geometric centers of the log-time grid."""
    g = cfg["grid"]
    edges = np.geomspace(t_end * float(g["span_ratio"]), t_end, int(g["n_points"]) + 1)
    return edges, np.sqrt(edges[:-1] * edges[1:])


def _bin(t, y, edges):
    """Mean of y and of ln t per bin; empty bins give NaN."""
    idx = np.searchsorted(edges, t, side="right") - 1
    keep = (idx >= 0) & (idx < edges.size - 1)
    idx, ys, lt = idx[keep], y[keep], np.log(t[keep])
    n = edges.size - 1
    cnt = np.bincount(idx, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ym = np.bincount(idx, weights=ys, minlength=n) / cnt
        ltm = np.bincount(idx, weights=lt, minlength=n) / cnt
    return ym, ltm, cnt


def _to_centers(ym, ltm, cnt, centers):
    """Bin means sit at their mean ln t; interpolate to the bin centers (fills empty bins)."""
    ok = cnt > 0
    if ok.sum() < 2:
        raise ValueError("record has fewer than 2 populated bins")
    return np.interp(np.log(centers), ltm[ok], ym[ok])


def make_features(proc: Processed, cfg: dict = _CFG) -> dict:
    edges, centers = time_grid(proc.t_end, cfg)
    dT_m, lt_m, cnt = _bin(proc.t, proc.dT, edges)
    P_m, _, _ = _bin(proc.t, proc.P, edges)
    dT_c = _to_centers(dT_m, lt_m, cnt, centers)
    P_c = _to_centers(P_m, lt_m, cnt, centers)
    theta = PI15 * proc.a * dT_c / P_c
    nref = int(cfg["grid"]["ref_bins"])
    theta_ref = float(np.mean(theta[-nref:]))
    theta_ref = max(theta_ref, 1e-9)
    dlnt = float(np.log(centers[1] / centers[0]))
    d = cfg["derivative"]
    dtheta = savgol_filter(theta, int(d["savgol_window"]), int(d["savgol_order"]), deriv=1, delta=dlnt,
                           mode="interp")
    seq = np.stack([theta / theta_ref, dtheta / theta_ref, np.log1p(cnt)]).astype(np.float32)
    scal = np.array([np.log(theta_ref), np.log(proc.t_end), np.log(proc.P0), float(np.mean(dT_c[-nref:])),
                     proc.thickness * 1e3, proc.radius * 1e3, float(proc.current_range)], dtype=np.float32)
    return {"seq": seq, "scalars": scal, "t_grid": centers, "theta": theta}


def targets(prm: dict) -> np.ndarray:
    return np.array([np.log(prm["k"]), np.log(prm["alpha"]), np.log(prm["G_c"])], dtype=np.float32)
