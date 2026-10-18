"""Phase 0 smoke tests: package imports, configs load, device helper works."""

from pathlib import Path

import pytest

from tpslib.config import CONFIG_DIR, load_sensor, load_yaml
from tpslib.runtime import get_device


def test_all_configs_load():
    for name in ["sensor", "device", "sim_numerics", "prior_v1", "dataset_v1", "train_m2"]:
        assert isinstance(load_yaml(CONFIG_DIR / f"{name}.yaml"), dict)


def test_scientific_notation_is_float(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text("a: 0.8e6\nb: 3.45e6\nc: 6.4e-3\nd: 1e-6\ne: 12\nf: name\n", encoding="utf-8")
    d = load_yaml(f)
    assert d == {"a": 0.8e6, "b": 3.45e6, "c": 6.4e-3, "d": 1e-6, "e": 12, "f": "name"}
    assert isinstance(d["a"], float) and isinstance(d["e"], int)


def test_no_string_numbers_in_configs():
    def walk(x, where):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{where}.{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{where}[{i}]")
        elif isinstance(x, str):
            try:
                float(x)
            except ValueError:
                return
            raise AssertionError(f"numeric string in {where}: {x!r}")

    for p in CONFIG_DIR.glob("*.yaml"):
        walk(load_yaml(p), p.name)


def test_sensor_matches_project_rules():
    s = load_sensor()
    assert s.n_rings == 20
    assert s.outer_radius == pytest.approx(6.4e-3)
    assert s.ring_radii()[-1] == pytest.approx(6.4e-3)
    assert s.ring_radii()[0] == pytest.approx(0.32e-3)


def test_unit_guard_rejects_mm():
    d = load_yaml(CONFIG_DIR / "sensor.yaml")
    d["spiral"]["outer_radius"] = 6.4  # mm by mistake
    with pytest.raises(ValueError):
        type(load_sensor()).from_dict(d)


def test_get_device():
    torch = pytest.importorskip("torch")
    dev = get_device()
    assert dev.type in ("cuda", "cpu")
    assert get_device("cpu").type == "cpu"
    assert torch.zeros(2, device=dev).sum().item() == 0.0
