"""Runtime helpers: compute device selection and project paths (local or Colab/Drive)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repository root: src/tpslib/runtime.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_device(prefer: str | None = None):
    """Return a torch.device: CUDA if available, else CPU.

    `prefer` ("cuda" / "cpu") or the TPS_DEVICE env var can force a choice.
    Forcing "cuda" without a GPU raises instead of silently falling back.
    """
    import torch

    choice = (prefer or os.environ.get("TPS_DEVICE", "auto")).lower()
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if choice != "auto":
        raise ValueError(f"Unknown device choice: {choice!r}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device) -> str:
    """Human-readable one-line description of a torch.device."""
    import torch

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx} {props.name} ({props.total_memory / 2**30:.1f} GB)"
    return f"cpu ({os.cpu_count()} logical cores)"


def is_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def data_root() -> Path:
    """Root for generated data. Override with TPS_DATA_ROOT (e.g. a Drive folder on Colab)."""
    return Path(os.environ.get("TPS_DATA_ROOT", PROJECT_ROOT / "data"))


def runs_root() -> Path:
    """Root for training runs. Override with TPS_RUNS_ROOT."""
    return Path(os.environ.get("TPS_RUNS_ROOT", PROJECT_ROOT / "runs"))
