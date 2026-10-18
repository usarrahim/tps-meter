"""r-z mesh and thin-film layer layout for the TPS sensor sandwich.

Coordinates: r from the sensor axis, z measured away from the sensor plane into each sample
(the bottom and top samples each get their own z mesh starting at the plane).

Two radial resolutions (non-conforming, nested):
- Fine columns, aligned with the traces, used by the film nodes and by the "near zone" of each
  sample (z < z_split). Trace edges carry a thin boundary layer (lateral leakage from the copper
  into the adjacent adhesive/PI, width ~ sqrt(k_layer t_layer / U) ~ 20 um), so cells next to
  every trace edge are `edge_cell` wide and grow geometrically away from it.
- Coarse columns (unions of fine columns) for the "far zone" (z > z_split), where the field only
  varies on the scale of the sensor radius. This keeps the far field accurate (small z growth)
  at low cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def geometric_segment(x0: float, x1: float, d0: float, growth: float) -> np.ndarray:
    """Faces from x0 to x1 (inclusive) with first cell ~d0 growing by `growth`, ending exactly at x1."""
    L = x1 - x0
    if L <= 0:
        return np.array([x0])
    if growth <= 1.0 + 1e-12 or d0 >= L:
        n = max(1, int(np.ceil(L / d0)))
        return np.linspace(x0, x1, n + 1)
    n = max(1, int(np.ceil(np.log(1.0 + L * (growth - 1.0) / d0) / np.log(growth))))
    d = L * (growth - 1.0) / (growth**n - 1.0)
    sizes = d * growth ** np.arange(n)
    faces = x0 + np.concatenate([[0.0], np.cumsum(sizes)])
    faces[-1] = x1
    return faces


def edge_graded(x0: float, x1: float, d_edge: float, growth: float, n_min: int,
                left: bool = True, right: bool = True) -> np.ndarray:
    """Faces on [x0, x1] with cells ~d_edge at the chosen ends growing by `growth` towards the
    middle; falls back to n_min uniform cells if that would give fewer cells."""
    sides = int(left) + int(right)
    if sides == 0 or d_edge <= 0:
        return np.linspace(x0, x1, n_min + 1)
    sizes = np.diff(geometric_segment(0.0, (x1 - x0) / sides, d_edge, growth))
    if sides * sizes.size < n_min:
        return np.linspace(x0, x1, n_min + 1)
    if left and right:
        sizes = np.concatenate([sizes, sizes[::-1]])
    elif right:
        sizes = sizes[::-1]
    faces = x0 + np.concatenate([[0.0], np.cumsum(sizes)])
    faces[-1] = x1
    return faces


def radial_faces(a, m, b, R_film, R_sample, n_center, n_trace, n_gap, r_growth,
                 edge_cell=0.0, edge_growth=1.5, trace_growth=None):
    """Fine, trace-aligned radial faces. Returns (faces, trace_ring)."""
    pitch = a / m
    if b >= pitch:
        raise ValueError("trace width must be smaller than the ring pitch")
    eg = lambda x0, x1, n, l, r, g=edge_growth: edge_graded(x0, x1, edge_cell, g, n, l, r)
    g_tr = edge_growth if trace_growth is None else trace_growth
    faces = [eg(0.0, pitch - b / 2, n_center, False, True)]
    ring = [np.full(faces[0].size - 1, -1)]
    for i in range(1, m + 1):
        ri = i * pitch
        tr = eg(ri - b / 2, ri + b / 2, n_trace, True, True, g_tr)
        faces.append(tr[1:])
        ring.append(np.full(tr.size - 1, i - 1))
        if i < m:
            gp = eg(ri + b / 2, ri + pitch - b / 2, n_gap, True, True)
            faces.append(gp[1:])
            ring.append(np.full(gp.size - 1, -1))
    d_out = edge_cell if edge_cell > 0 else (pitch - b) / n_gap
    edge = a + b / 2
    start = edge
    if R_film > edge:
        seg = geometric_segment(edge, R_film, d_out, r_growth)
        faces.append(seg[1:])
        ring.append(np.full(seg.size - 1, -1))
        d_out, start = seg[-1] - seg[-2], R_film
    if R_sample > start:
        seg = geometric_segment(start, R_sample, d_out, r_growth)
        faces.append(seg[1:])
        ring.append(np.full(seg.size - 1, -1))
    return np.concatenate(faces), np.concatenate(ring).astype(int)


def coarsen_faces(fine: np.ndarray, d_min: float, growth: float, r_edge: float) -> np.ndarray:
    """Subset of fine faces (greedy) with cell widths >= max(d_min, (growth-1) (r - r_edge)+).

    The far field varies on the sensor scale, so the coarse cells are uniform (d_min) under the
    sensor and grow only outside it.
    """
    target = lambda r: max(d_min, (growth - 1.0) * max(r - r_edge, 0.0))
    out = [fine[0]]
    for f in fine[1:-1]:
        if f - out[-1] >= target(out[-1]):
            out.append(f)
    if fine[-1] - out[-1] < 0.5 * target(out[-1]) and len(out) > 1:
        out.pop()
    out.append(fine[-1])
    return np.array(out)


def z_faces(H: float, dz0: float, g_near: float, z_split: float, g_far: float) -> tuple[np.ndarray, int]:
    """z faces of one sample and the number of near-zone cells (z < ~z_split)."""
    if z_split <= 0 or z_split >= H:
        f = geometric_segment(0.0, H, dz0, g_near)
        return f, f.size - 1
    sizes, z = [], 0.0
    while z < z_split:
        d = dz0 * g_near ** len(sizes)
        sizes.append(d)
        z += d
    if z >= H:
        f = geometric_segment(0.0, H, dz0, g_near)
        return f, f.size - 1
    near = np.concatenate([[0.0], np.cumsum(sizes)])
    far = geometric_segment(near[-1], H, sizes[-1] * g_far, g_far)
    return np.concatenate([near, far[1:]]), len(sizes)


@dataclass
class Layer:
    name: str
    thickness: float        # [m]
    k: np.ndarray           # [W/m/K] per film column
    rho_cp: np.ndarray      # [J/m^3/K] per film column
    heater: bool = False


@dataclass
class SampleMesh:
    z_faces: np.ndarray     # distance from the sensor plane
    n_near: int             # z cells in the near zone (fine columns); the rest use coarse columns

    @property
    def dz(self) -> np.ndarray:
        return np.diff(self.z_faces)


@dataclass
class Mesh:
    r_faces: np.ndarray         # fine radial faces (nr+1,)
    r_faces_coarse: np.ndarray  # coarse radial faces, a subset of r_faces
    trace_ring: np.ndarray      # (nr,) ring index 0..m-1 for trace columns, -1 elsewhere
    n_film: int                 # fine columns 0..n_film-1 are covered by the film
    bottom: SampleMesh
    top: SampleMesh
    layers: list[Layer] = field(default_factory=list)

    @property
    def nr(self) -> int:
        return self.r_faces.size - 1

    @property
    def r_c(self) -> np.ndarray:
        return 0.5 * (self.r_faces[:-1] + self.r_faces[1:])

    @property
    def annulus(self) -> np.ndarray:
        """Fine column cross-section areas pi (r_o^2 - r_i^2) [m^2]."""
        return np.pi * (self.r_faces[1:] ** 2 - self.r_faces[:-1] ** 2)

    @property
    def is_trace(self) -> np.ndarray:
        return self.trace_ring >= 0

    @property
    def fine_to_coarse(self) -> np.ndarray:
        """Coarse column index of every fine column."""
        return np.searchsorted(self.r_faces_coarse, self.r_c) - 1


def build_mesh(sensor_cfg: dict, numerics: dict, sample_radius: float, thickness_bottom: float,
               thickness_top: float, materials_override: dict | None = None,
               layer_scale: dict | None = None, layers_on: bool = True, refine: int = 1) -> Mesh:
    """Build the mesh and film layers.

    sensor_cfg : raw sensor.yaml dict
    materials_override : e.g. {"polyimide": {"k": 0.15}}
    layer_scale : thickness multipliers per layer name, e.g. {"polyimide_base": 1.1}
    layers_on=False : no film; a single zero-thickness, zero-capacity plane node carries the
                      source (analytic-limit configuration).
    refine : 2 halves every cell size / growth exponent (convergence test).
    """
    sp = sensor_cfg["spiral"]
    a, m, b = float(sp["outer_radius"]), int(sp["n_rings"]), float(sp["trace_width"])
    nm = numerics["mesh"]
    f = int(refine)
    root = lambda g: float(g) ** (1.0 / f)
    R_film = float(sensor_cfg["film"]["outer_radius"]) if layers_on else sample_radius
    R_film = min(R_film, sample_radius)
    r_f, ring = radial_faces(
        a, m, b, R_film, sample_radius,
        n_center=int(nm["n_center"]) * f, n_trace=int(nm["n_trace"]) * f, n_gap=int(nm["n_gap"]) * f,
        r_growth=root(nm["r_growth"]), edge_cell=float(nm.get("edge_cell", 0.0)) / f,
        edge_growth=root(nm.get("edge_growth", 1.5)),
        trace_growth=root(nm["trace_growth"]) if "trace_growth" in nm else None,
    )
    r_coarse = coarsen_faces(r_f, float(nm["coarse_dr"]) / f, root(nm["coarse_growth"]), a)
    dz0 = float(nm["dz_min"]) / f
    zs = float(nm["z_split"])
    zb, nb = z_faces(thickness_bottom, dz0, root(nm["z_growth"]), zs, root(nm["z_growth_far"]))
    zt, nt = z_faces(thickness_top, dz0, root(nm["z_growth"]), zs, root(nm["z_growth_far"]))
    r_c = 0.5 * (r_f[:-1] + r_f[1:])
    n_film = int(np.searchsorted(r_c, R_film))
    mesh = Mesh(r_faces=r_f, r_faces_coarse=r_coarse, trace_ring=ring, n_film=n_film,
                bottom=SampleMesh(zb, nb), top=SampleMesh(zt, nt))

    if not layers_on:
        z = np.zeros(n_film)
        mesh.layers = [Layer("plane", 0.0, z.copy(), z.copy(), heater=True)]
        return mesh

    mats = {k: dict(v) for k, v in sensor_cfg["materials"].items()}
    for name, props in (materials_override or {}).items():
        mats[name].update(props)
    trace = ring[:n_film] >= 0
    for spec in sensor_cfg["film"]["stack"]:
        t = float(spec["thickness"]) * float((layer_scale or {}).get(spec["name"], 1.0))
        main = mats[spec["material"]]
        k = np.full(n_film, float(main["k"]))
        rc = np.full(n_film, float(main["rho_cp"]))
        if "gap_material" in spec:
            gap = mats[spec["gap_material"]]
            k[~trace] = float(gap["k"])
            rc[~trace] = float(gap["rho_cp"])
        mesh.layers.append(Layer(spec["name"], t, k, rc, heater=bool(spec.get("heater", False))))
    if sum(l.heater for l in mesh.layers) != 1:
        raise ValueError("exactly one film layer must be the heater")
    return mesh
