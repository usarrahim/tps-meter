"""Pack code + configs (no data, no runs, no venv) into dist/tps-meter-code.zip for Google Drive/Colab.

Usage: python scripts/pack_for_colab.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "configs", "scripts", "tests", "notebooks", "docs"]
INCLUDE_FILES = ["pyproject.toml", "README.md"]
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache"}


def main() -> None:
    """Write dist/tps-meter-code.zip."""
    out = ROOT / "dist" / "tps-meter-code.zip"
    out.parent.mkdir(exist_ok=True)
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in INCLUDE_FILES:
            p = ROOT / name
            if p.exists():
                zf.write(p, p.relative_to(ROOT).as_posix())
                n += 1
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and not SKIP_PARTS.intersection(p.parts) and "egg-info" not in str(p):
                    zf.write(p, p.relative_to(ROOT).as_posix())
                    n += 1
    print(f"{n} files -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
