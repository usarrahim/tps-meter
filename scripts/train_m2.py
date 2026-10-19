"""Train M2 (1D ResNet, beta-NLL). One run per seed; re-running the same command resumes unfinished runs.

Examples
  # 5-epoch trial on the pilot file (development only: 90/10 split of pilot.h5)
  python scripts/train_m2.py --dev-pilot --seeds 1 --max-epochs 5 --tag trial
  # one seed, full data
  python scripts/train_m2.py --seeds 1
  # 5-seed ensemble
  python scripts/train_m2.py --seeds 1 2 3 4 5
  # ablation: k band 1.5-2.5 removed from train/val
  python scripts/train_m2.py --seeds 1 --exclude-k-band 1.5 2.5 --tag ablation
  # data-size study
  python scripts/train_m2.py --seeds 1 --max-train 10000 --tag n10k
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path

from tpslib.config import load_yaml
from tpslib.data.generate import dataset_dir
from tpslib.runtime import runs_root
from tpslib.train.trainer import train


def main() -> None:
    """Train one run per seed (resuming unfinished runs) and print the best epochs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_m2.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: train.seeds from the config")
    ap.add_argument("--tag", default=None, help="run name tag (default: date); same tag + seed resumes")
    ap.add_argument("--dev-pilot", action="store_true", help="use pilot.h5 split 90/10 (code testing only)")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--exclude-k-band", type=float, nargs=2, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override (e.g. small pilot trials)")
    ap.add_argument("--max-asym", type=float, default=None,
                    help="keep only samples with contact asymmetry h_max/h_min <= this (train and val)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = copy.deepcopy(cfg)
    if args.exclude_k_band:
        cfg["data"]["exclude_k_band"] = list(args.exclude_k_band)
    if args.max_train:
        cfg["data"]["max_train"] = int(args.max_train)
    if args.max_asym:
        cfg["data"]["max_contact_asymmetry"] = float(args.max_asym)
    if args.batch_size:
        cfg["train"]["batch_size"] = int(args.batch_size)
    data_dir = dataset_dir(load_yaml(cfg["data"]["dataset"]))
    tag = args.tag or dt.date.today().strftime("%Y%m%d")
    seeds = args.seeds or [int(s) for s in cfg["train"]["seeds"]]
    results = {}
    for seed in seeds:
        run_dir = Path(runs_root()) / f"{tag}_{cfg['name']}_seed{seed}"
        print(f"=== {run_dir}", flush=True)
        kw = dict(train_file="pilot.h5", dev_split=0.9) if args.dev_pilot else {}
        s = train(cfg, data_dir, run_dir, seed, resume=not args.no_resume, max_epochs=args.max_epochs,
                  num_workers=args.num_workers, log=lambda m: print(m, flush=True), **kw)
        results[seed] = {k: s[k] for k in ("best_epoch", "epochs_run")}
        results[seed].update({k: v for k, v in s["best"].items() if "rel_err_ln" in k or "cov95" in k})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
