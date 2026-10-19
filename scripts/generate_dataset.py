"""Generate synthetic dataset splits (clean curves). Resumable: re-run the same command after an interruption.

Usage:
  python scripts/generate_dataset.py --config configs/dataset_v1.yaml --splits pilot
  python scripts/generate_dataset.py --config configs/dataset_v1.yaml --splits train val test test_interp test_faulty
"""

from __future__ import annotations

import os

# One BLAS thread per process (parallelism comes from the process pool). Inherited by spawned workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import shutil
from pathlib import Path

from tpslib.config import load_yaml
from tpslib.data.generate import dataset_dir, run_split, update_manifest


def main() -> None:
    """Generate the requested splits (resumable) and update the manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dataset_v1.yaml")
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--n", type=int, default=None, help="override sample count (debugging)")
    ap.add_argument("--mirror", default=None,
                    help="directory (e.g. on Google Drive) that keeps a copy of finished shards/splits; "
                         "they are restored from it when the run is restarted")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out_dir = dataset_dir(cfg)
    for split in args.splits:
        sc = cfg["pilot"] if split == "pilot" else cfg["splits"][split]
        n = args.n or int(sc["n"])
        mirror = Path(args.mirror) if args.mirror else None
        stats = run_split(split, n, int(sc["seed"]), out_dir, args.workers, int(cfg["shard_size"]),
                          k_band=sc.get("k_band"), faulty=split == "test_faulty", mirror=mirror)
        if mirror is not None and (mirror / "manifest.json").exists() and not (out_dir / "manifest.json").exists():
            shutil.copyfile(mirror / "manifest.json", out_dir / "manifest.json")
        update_manifest(out_dir, cfg, split, stats)
        if mirror is not None:
            shutil.copyfile(out_dir / "manifest.json", mirror / "manifest.json")
        print(json.dumps({split: stats}, indent=2, default=float))


if __name__ == "__main__":
    main()
