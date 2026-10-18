"""Where do the errors come from? Break the validation error of a trained run down by parameter bands.

Usage: python scripts/diagnose.py --run n10k_m2_seed1 [--split val] [--num-workers 8]
Writes diagnose_<split>.csv / .json / .png into the run folder and prints the tables.
Uses the validation split only (the test splits stay locked until Phase 6).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tpslib.config import load_yaml
from tpslib.data.dataset import TPSDataset
from tpslib.data.generate import dataset_dir
from tpslib.eval.metrics import breakdown, calibration_floor, coverage, rel_errors, summarize
from tpslib.eval.predict import load_run, predict
from tpslib.runtime import runs_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run folder name under the runs root (or a path)")
    ap.add_argument("--split", default="val", choices=["val", "pilot"])
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run) if Path(args.run).exists() else Path(runs_root()) / args.run
    model, norm, cfg = load_run(run)
    data = dataset_dir(load_yaml(cfg["data"]["dataset"])) / f"{args.split}.h5"
    ds = TPSDataset(data, train=False, base_seed=2002)
    pred = predict(model, norm, ds, num_workers=args.num_workers)
    p = {k: v[pred["index"]] for k, v in ds.params.items()}
    err = rel_errors(pred)

    P0, sps = p["P0"], p["sps"]
    asym = np.maximum(p["h_c_bottom"], p["h_c_top"]) / np.minimum(p["h_c_bottom"], p["h_c_top"])
    by = {
        "k [W/mK]": (p["k"], [(0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 30.01)]),
        "G_c [W/m2K]": (p["G_c"], [(0, 2000), (2000, 4000), (4000, 8000), (8000, 1e9)]),
        "tau_end": (p["tau_end"], [(0, 0.7), (0.7, 0.9), (0.9, 1.05), (1.05, 9)]),
        "kalinlik [mm]": (p["sample_thickness"] * 1e3, [(8, 12), (12, 16), (16, 20), (20, 25.01)]),
        "sure derinlikle sinirli": (p["t_end_limited_by_depth"], [0, 1]),
        "guc 1 W'ta kirpik": ((P0 >= 0.9999).astype(int), [0, 1]),
        "guc 8 mW'ta kirpik": ((P0 <= 0.008 * 1.0001).astype(int), [0, 1]),
        "SPS": (sps, sorted(set(sps.tolist()))),
        "temas asimetrisi h_max/h_min": (asym, [(1, 2), (2, 4), (4, 10), (10, 30), (30, 101)]),
    }
    table = breakdown(pred, p, by)
    # cross-table: k band x contact asymmetry (median / p95 |k error| %)
    cross = []
    for kb in [(0.05, 2), (2, 5), (5, 30.01)]:
        for ab in [(1, 4), (4, 10), (10, 101)]:
            sel = (p["k"] >= kb[0]) & (p["k"] < kb[1]) & (asym >= ab[0]) & (asym < ab[1])
            if sel.any():
                s_ = summarize(err["k"][sel])
                cross.append({"k": str(kb), "asimetri": str(ab), "n": s_["n"], "k_med_pct": s_["med_pct"],
                              "k_p95_pct": s_["p95_pct"], "k_bias_pct": s_["bias_pct"]})
    cross = pd.DataFrame(cross)
    floor = calibration_floor(ds.params, power_error_rel=0.005)
    overall = {t: summarize(err[t]) for t in ("k", "alpha", "G_c")}
    cov = {"k_cov68": coverage(pred, 0, 1.0), "k_cov95": coverage(pred, 0, 1.96),
           "alpha_cov95": coverage(pred, 1, 1.96)}
    worst = np.argsort(-np.abs(err["k"]))[:15]
    cols = ["k", "alpha", "G_c", "h_c_bottom", "h_c_top", "sample_thickness", "tau_end", "P0", "sps", "t_end"]
    wt = pd.DataFrame({c: p[c][worst] for c in cols})
    wt.insert(0, "k_err_pct", err["k"][worst] * 100)

    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(f"run: {run.name}  split: {args.split}  n={len(ds)}")
    print("overall:", json.dumps(overall), json.dumps(cov))
    print("calibration floor (unremovable):", json.dumps(floor))
    print(table.round(2).to_string(index=False))
    print("\nk bandi x temas asimetrisi:")
    print(cross.round(2).to_string(index=False))
    print("\nworst 15 (k):")
    print(wt.round(4).to_string(index=False))

    table.to_csv(run / f"diagnose_{args.split}.csv", index=False)
    cross.to_csv(run / f"diagnose_{args.split}_k_x_asymmetry.csv", index=False)
    (run / f"diagnose_{args.split}.json").write_text(json.dumps(
        {"overall": overall, "coverage": cov, "calibration_floor": floor}, indent=2), encoding="utf-8")
    from tpslib.eval.report import plt, save, style
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    sc = axs[0].scatter(p["k"], err["k"] * 100, s=3, c=np.log10(p["G_c"]), cmap="viridis", alpha=0.6)
    fig.colorbar(sc, ax=axs[0]).set_label("log10 G_c")
    axs[0].set_xscale("log"); axs[0].set_xlabel("k [W/m·K]"); axs[0].set_ylabel("k hatası [%]")
    axs[0].set_ylim(-40, 40); style(axs[0], "k hatası – k (renk: G_c)")
    axs[1].scatter(p["tau_end"], err["k"] * 100, s=3, alpha=0.5)
    axs[1].set_xlabel("τ_son"); axs[1].set_ylim(-40, 40); style(axs[1], "k hatası – τ_son")
    axs[2].scatter(p["k"], err["alpha"] * 100, s=3, alpha=0.5)
    axs[2].set_xscale("log"); axs[2].set_xlabel("k [W/m·K]"); axs[2].set_ylabel("α hatası [%]")
    axs[2].set_ylim(-60, 60); style(axs[2], "α hatası – k")
    save(fig, run / f"diagnose_{args.split}.png")


if __name__ == "__main__":
    main()
