"""YAML -> dataclass loading with basic SI unit sanity checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tpslib.runtime import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads 1e6 / 0.8e6 as floats (YAML 1.1 would return strings)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                   |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                   |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                   |[-+]?\.(?:inf|Inf|INF)
                   |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file. Relative paths resolve against the project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open("r", encoding="utf-8") as f:
        data = yaml.load(f, Loader=_Loader)
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def _check_range(name: str, value: float, lo: float, hi: float) -> float:
    """Guard against unit mistakes (e.g. mm written instead of m)."""
    value = float(value)
    if not (lo <= value <= hi):
        raise ValueError(f"{name}={value} outside plausible SI range [{lo}, {hi}] - check units")
    return value


@dataclass(frozen=True)
class SensorConfig:
    """Sensor geometry and electrical constants from configs/sensor.yaml (SI units)."""
    n_rings: int
    outer_radius: float      # [m]
    trace_width: float       # [m]
    copper_thickness: float  # [m]
    R0: float                # [ohm]
    alpha_R: float           # [1/K]
    raw: dict

    @classmethod
    def from_dict(cls, d: dict) -> "SensorConfig":
        """Build from the raw YAML dict; rejects values outside plausible SI ranges (unit mistakes)."""
        if d.get("units") != "SI":
            raise ValueError("sensor config must declare units: SI")
        s, e = d["spiral"], d["electrical"]
        return cls(
            n_rings=int(s["n_rings"]),
            outer_radius=_check_range("outer_radius", s["outer_radius"], 1e-4, 0.1),
            trace_width=_check_range("trace_width", s["trace_width"], 1e-6, 1e-2),
            copper_thickness=_check_range("copper_thickness", s["copper_thickness"], 1e-7, 1e-3),
            R0=_check_range("R0", e["R0"], 0.01, 1e3),
            alpha_R=_check_range("alpha_R", e["alpha_R"], 1e-4, 1e-2),
            raw=d,
        )

    def ring_radii(self):
        """r_i = i * a / m, i = 1..m [m]."""
        import numpy as np

        return np.arange(1, self.n_rings + 1) * self.outer_radius / self.n_rings


def load_sensor(path: str | Path = CONFIG_DIR / "sensor.yaml") -> SensorConfig:
    """Load and validate configs/sensor.yaml."""
    return SensorConfig.from_dict(load_yaml(path))
