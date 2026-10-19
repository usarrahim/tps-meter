"""Phase 6: final evaluation on the locked test splits, M0 comparison, calibration, interpolation
(ablation), faulty-measurement detection, compact model, ONNX export and the model card.

The test splits are used ONCE here; all thresholds come from the validation split.

Usage (Colab):
  python scripts/evaluate.py --ensemble-tag v1sym --seeds 1 2 3 4 5 --ablation ablation_m2_seed1 \
      --compact-tag v1c --compact-seeds 1 --max-asym 10
Writes <runs root>/../reports/phase6/ (figures, CSV, summary.json, MODEL_CARD.md, onnx/) and
<runs root>/../phase6_bundle.zip with all of it.

  python scripts/evaluate.py --trust-only ...
Trust checks v2 and the variance scale only (val, test, test_faulty; no ablation, compact or ONNX).
This is a SECOND use of the test splits (the first was Phase 6) and is labelled as such everywhere.
Writes <runs root>/../reports/phase6_trust/ and <runs root>/../phase6_trust_bundle.zip.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as dt
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tpslib.config import load_yaml
from tpslib.data.dataset import TPSDataset, contact_asymmetry
from tpslib.data.features import CHANNELS, SCALARS
from tpslib.data.generate import dataset_dir
from tpslib.data.prior import FAULTS
from tpslib.eval.m0_batch import run_m0
from tpslib.eval.metrics import K_BANDS, coverage, rel_errors, summarize
from tpslib.eval.predict import ensemble, load_run, predict_members
from tpslib.eval.report import BAND, INK2, MARKERS, SERIES, plt, save, style
from tpslib.eval.trust import (TARGET_NAMES, fit_thresholds, fit_thresholds_v2, fit_variance_scale, flags,
                               flags_v2, interval_coverage, load_trust_config, trust_scores)
from tpslib.models.export import EnsembleWithNorm, export_onnx, verify_onnx
from tpslib.models.resnet1d import count_parameters
from tpslib.runtime import runs_root

TARGET_MED, TARGET_P95 = 1.0, 4.0
T0 = time.perf_counter()
SPLITS = ("val", "test", "test_interp", "test_faulty")
TRUST_SPLITS = ("val", "test", "test_faulty")
SECOND_USE = ("SECOND USE: the test sets were first used in Phase 6. This run only reports the new trust checks "
              "and the variance scale, both chosen on val; no model or threshold choice looked at the test sets.")


def band_table(err: dict, k: np.ndarray, sel: np.ndarray, label: str) -> list[dict]:
    rows = []
    for lo, hi in [*K_BANDS, (0.05, 30.01)]:
        m = sel & (k >= lo) & (k < hi)
        ek, ea = summarize(err["k"][m]), summarize(err["alpha"][m])
        rows.append({"model": label, "k_band": f"{lo}-{min(hi, 30)}" if hi < 30.01 else "all", "n": ek["n"],
                     "k_med_pct": ek["med_pct"], "k_p95_pct": ek["p95_pct"], "k_bias_pct": ek["bias_pct"],
                     "a_med_pct": ea["med_pct"], "a_p95_pct": ea["p95_pct"]})
    return rows


def m0_errors(m0: dict, p: dict) -> dict:
    return {"k": m0["k"] / p["k"] - 1, "alpha": m0["alpha"] / p["alpha"] - 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-tag", default="v1sym")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--ablation", default="ablation_m2_seed1")
    ap.add_argument("--compact-tag", default="v1c")
    ap.add_argument("--compact-seeds", type=int, nargs="*", default=[1])
    ap.add_argument("--max-asym", type=float, default=10.0)
    ap.add_argument("--num-workers", type=int, default=os.cpu_count())
    ap.add_argument("--data-dir", default=None, help="default: dataset_v1 location")
    ap.add_argument("--split-files", nargs="*", default=None, help="override, e.g. test=pilot.h5 (dry runs)")
    ap.add_argument("--out", default=None, help="default: <runs root>/../reports/phase6[_trust]")
    ap.add_argument("--trust-only", action="store_true",
                    help="second use of the test splits: trust checks v2 + variance scale only")
    args = ap.parse_args()
    t_start = T0

    root = Path(runs_root())
    name = "phase6_trust" if args.trust_only else "phase6"
    out = Path(args.out) if args.out else root.parent / "reports" / name
    out.mkdir(parents=True, exist_ok=True)
    if not args.trust_only:
        (out / "onnx").mkdir(exist_ok=True)
    tcfg = load_trust_config()
    runs = [root / f"{args.ensemble_tag}_m2_seed{s}" for s in args.seeds]
    loaded = [load_run(r) for r in runs]
    cfg = loaded[0][2]
    data_dir = Path(args.data_dir) if args.data_dir else dataset_dir(load_yaml(cfg["data"]["dataset"]))
    files = {s: f"{s}.h5" for s in SPLITS}
    for kv in args.split_files or []:
        k_, v_ = kv.split("="); files[k_] = v_
    log = lambda m: print(f"[{time.perf_counter() - t_start:6.0f}s] {m}", flush=True)
    nw = args.num_workers

    # --- predictions -----------------------------------------------------------------------------
    ds, ens, params, m0 = {}, {}, {}, {}
    for split in (TRUST_SPLITS if args.trust_only else SPLITS):
        path = data_dir / files[split]
        if not path.exists():
            log(f"skip {split}: {path} missing"); continue
        ds[split] = TPSDataset(path, train=False, base_seed=2002 if split == "val" else 3003)
        ens[split] = ensemble(predict_members([(m, nm) for m, nm, _ in loaded], ds[split], num_workers=nw))
        params[split] = {k: v[ens[split]["index"]] for k, v in ds[split].params.items()}
        log(f"M2 ensemble on {split}: n={len(ds[split])}")
        m0[split] = run_m0(ds[split], workers=nw)
        log(f"M0 on {split}: ok={m0[split]['ok'].mean():.3f}")

    def in_domain(p):
        return contact_asymmetry(p) <= args.max_asym

    summary = {"created": dt.datetime.now().isoformat(timespec="seconds"), "ensemble": [r.name for r in runs],
               "n_params_member": count_parameters(loaded[0][0]), "max_contact_asymmetry": args.max_asym,
               "targets": {"k_med_pct": TARGET_MED, "k_p95_pct": TARGET_P95},
               "test_use": SECOND_USE if args.trust_only else "first (Phase 6)"}
    p, e = params["test"], ens["test"]
    dom = in_domain(p)
    dv = in_domain(params["val"])

    if args.trust_only:
        calibration_section(ens, dv, dom, tcfg, summary, out, second_use=True)
        if "test_faulty" in ens:
            trust_section(ens, m0, params, dv, dom, tcfg, summary, out, second_use=True)
        finish(summary, out, name, log)
        write_trust_report(summary, out)
        return

    # --- test: accuracy per k band, M2 vs M0 -----------------------------------------------------------
    err, err0 = rel_errors(e), m0_errors(m0["test"], p)
    rows = band_table(err, p["k"], dom, "M2 ensemble") + band_table(err0, p["k"], dom & m0["test"]["ok"], "M0")
    rows += band_table(err, p["k"], ~dom, "M2 (asymmetry > 10, fault domain)")
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "test_accuracy_by_band.csv", index=False)
    summary["test_accuracy"] = rows
    summary["test_targets_met"] = {r["k_band"]: bool(r["k_med_pct"] < TARGET_MED and r["k_p95_pct"] < TARGET_P95)
                                   for r in rows if r["model"] == "M2 ensemble"}
    summary["m0_fail_rate_test"] = float(1 - m0["test"]["ok"].mean())

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (lab, ee, c, mk) in zip(axs[:2], [("M2 ensemble", err, SERIES[0], "o"), ("M0 (classic)", err0, SERIES[1], "s")]):
        ax.axhspan(-TARGET_P95, TARGET_P95, color=BAND, zorder=0)
        ax.scatter(p["k"][dom], 100 * ee["k"][dom], s=3, color=c, alpha=0.5)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log"); ax.set_ylim(-30, 30)
        ax.set_xlabel("k [W/m·K]"); ax.set_ylabel("k relative error [%]")
        style(ax, f"{lab}, test (realistic contact); grey: ±{TARGET_P95:.0f}%")
    hmin = np.minimum(p["h_c_bottom"], p["h_c_top"])
    axs[2].scatter(hmin[dom], 100 * err["k"][dom], s=3, color=SERIES[0], alpha=0.5)
    axs[2].set_xscale("log"); axs[2].set_ylim(-30, 30); axs[2].axhline(0, color=INK2, lw=0.8)
    axs[2].set_xlabel("min(h_c bottom, h_c top) [W/m²·K]"); axs[2].set_ylabel("k relative error [%]")
    style(axs[2], "M2 error vs contact conductance")
    save(fig, out / "test_error_scatter.png")

    calibration_section(ens, dv, dom, tcfg, summary, out)

    # --- interpolation / ablation ---------------------------------------------------------------------
    if "test_interp" in ens and (root / args.ablation / "best.pt").exists():
        pi = params["test_interp"]
        di = in_domain(pi)
        abl = load_run(root / args.ablation)
        pa = predict_members([(abl[0], abl[1])], ds["test_interp"], num_workers=nw)[0]
        ei, ea = rel_errors(ens["test_interp"]), rel_errors(pa)
        pa_test = predict_members([(abl[0], abl[1])], ds["test"], num_workers=nw)[0]
        out_band = dom & ~((p["k"] >= 1.5) & (p["k"] <= 2.5))
        interp = {"main_on_band": summarize(ei["k"][di]), "ablation_on_band": summarize(ea["k"][di]),
                  "ablation_on_test_outside_band": summarize(rel_errors(pa_test)["k"][out_band]),
                  "main_on_test_outside_band": summarize(err["k"][out_band]),
                  "ablation_cov95_on_band": float(np.mean(np.abs(pa["mean"][di, 0] - pa["true"][di, 0])
                                                          <= 1.96 * pa["std"][di, 0]))}
        summary["interpolation"] = interp
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.scatter(pi["k"][di], 100 * ei["k"][di], s=4, color=SERIES[0], alpha=0.5, label="main model (saw the band)")
        ax.scatter(pi["k"][di], 100 * ea["k"][di], s=4, color=SERIES[1], alpha=0.5, label="ablation (band held out)")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("k [W/m·K]"); ax.set_ylabel("k error [%]")
        ax.legend(frameon=False); style(ax, "test_interp: k = 1.5–2.5 W/m·K band")
        save(fig, out / "interpolation.png")

    if "test_faulty" in ens:
        trust_section(ens, m0, params, dv, dom, tcfg, summary, out)

    # --- compact model ----------------------------------------------------------------------------------
    comp_runs = [root / f"{args.compact_tag}_m2c_seed{s}" for s in args.compact_seeds]
    comp_runs = [r for r in comp_runs if (r / "best.pt").exists()]
    if comp_runs:
        cl = [load_run(r) for r in comp_runs]
        ce = ensemble(predict_members([(m, nm) for m, nm, _ in cl], ds["test"], num_workers=nw))
        crow = band_table(rel_errors(ce), p["k"], dom, f"compact ({len(cl)} members)")
        diff = np.abs(np.expm1(ce["mean"][dom, 0] - e["mean"][dom, 0]))
        summary["compact"] = {"runs": [r.name for r in comp_runs], "n_params": count_parameters(cl[0][0]),
                              "accuracy": crow, "vs_ensemble_k_med_pct": float(np.median(diff) * 100),
                              "vs_ensemble_k_p99_pct": float(np.percentile(diff, 99) * 100)}
        pd.DataFrame(crow).to_csv(out / "compact_accuracy_by_band.csv", index=False)
        cm = EnsembleWithNorm([m for m, _, _ in cl], [nm for _, nm, _ in cl])
        export_onnx(cm, out / "onnx" / "m2c_compact.onnx", len(CHANNELS), len(SCALARS))

    # --- ONNX export of the ensemble ----------------------------------------------------------------
    em = EnsembleWithNorm([m for m, _, _ in loaded], [nm for _, nm, _ in loaded])
    onnx_path = export_onnx(em, out / "onnx" / "m2_ensemble.onnx", len(CHANNELS), len(SCALARS))
    feats = [ds["test"].features(j)[0] for j in range(64)]
    seq = np.stack([f["seq"] for f in feats]).astype(np.float32)
    sc = np.stack([f["scalars"] for f in feats]).astype(np.float32)
    summary["onnx"] = {"file": onnx_path.name, "size_MB": onnx_path.stat().st_size / 1e6,
                       "max_abs_diff_vs_pytorch": verify_onnx(em, onnx_path, seq, sc)}
    for r in runs:
        shutil.copy2(r / "normalizer.json", out / "onnx" / f"normalizer_{r.name}.json")

    if "trust_v2" in summary:
        shutil.copy2(out / "trust_v2.json", out / "onnx" / "trust_v2.json")
    write_model_card(summary, cfg, out)
    print(acc.round(2).to_string(index=False))
    finish(summary, out, name, log)


def finish(summary: dict, out: Path, name: str, log) -> None:
    summary["elapsed_s"] = time.perf_counter() - T0
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    bundle = shutil.make_archive(str(out.parent.parent / f"{name}_bundle"), "zip", out)
    log(f"done -> {out}  bundle: {bundle}")
    print(json.dumps({k: summary[k] for k in summary if k not in ("test_accuracy",)}, indent=1,
                     ensure_ascii=False, default=float)[:8000])


def calibration_section(ens: dict, dv, dom, tcfg: dict, summary: dict, out: Path, second_use=False) -> None:
    """Coverage of the predicted intervals on test, before and after the post-hoc variance scale that is
    fitted on val (realistic-contact domain) only."""
    ev, e = ens["val"], ens["test"]
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95, 0.99])
    i68, i95 = 6, 10
    scale = fit_variance_scale(ev["mean"][dv], ev["std"][dv], ev["true"][dv], tcfg["variance_scale"]["levels"])
    raw = interval_coverage(e["mean"][dom], e["std"][dom], e["true"][dom], levels)
    cal = interval_coverage(e["mean"][dom], e["std"][dom] * scale, e["true"][dom], levels)
    calv = interval_coverage(ev["mean"][dv], ev["std"][dv] * scale, ev["true"][dv], levels)
    cov = lambda c: {t: {"cov68": float(c[i68, i]), "cov95": float(c[i95, i])} for i, t in enumerate(TARGET_NAMES)}
    summary["calibration_test"] = cov(raw)
    summary["variance_scale"] = {
        "fitted_on": "val, contact asymmetry <= max_asym", "levels": tcfg["variance_scale"]["levels"],
        "scale": {t: float(scale[i]) for i, t in enumerate(TARGET_NAMES)},
        "val_scaled": cov(calv), "test_scaled": cov(cal)}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="ideal")
    for i, t in enumerate(TARGET_NAMES):
        ax.plot(levels, raw[:, i], MARKERS[i] + ":", color=SERIES[i], ms=4, alpha=0.6, label=f"{t} (raw)")
        ax.plot(levels, cal[:, i], MARKERS[i] + "-", color=SERIES[i], ms=5, label=f"{t} (std × {scale[i]:.2f})")
    ax.set_xlabel("expected coverage"); ax.set_ylabel("observed coverage"); ax.legend(frameon=False, fontsize=8)
    style(ax, "Uncertainty calibration (test" + (", second use" if second_use else "") + "; scale from val)")
    save(fig, out / "calibration.png")


FAULT_LABELS = {"air_gap_low_hc": "air gap", "thin_sample": "thin sample", "drift_jump": "drift jump",
                "contact_asymmetry_gt_max": "contact asymmetry > 10"}


def trust_section(ens: dict, m0: dict, params: dict, dv, dom, tcfg: dict, summary: dict, out: Path,
                  second_use=False) -> None:
    """Faulty-measurement detection with trust v1 (Phase 6) and v2 (range checks, spread of every target,
    shared false-alarm budget). Thresholds from val (realistic contact) only."""
    sub = lambda d, m: {k: v[m] for k, v in d.items()}
    dT_range = tcfg["dT_range_K"]
    sv = trust_scores(sub(ens["val"], dv), sub(m0["val"], dv))
    st, sf = trust_scores(ens["test"], m0["test"]), trust_scores(ens["test_faulty"], m0["test_faulty"])
    thr1 = fit_thresholds(sv, q=tcfg["v1"]["quantile"])
    thr2 = fit_thresholds_v2(sv, tcfg)
    kind = params["test_faulty"]["fault"]
    err_f = rel_errors(ens["test_faulty"])["k"]
    res = {}
    for ver, fl in (("v1", lambda s: flags(s, thr1, dT_range)), ("v2", lambda s: flags_v2(s, thr2, dT_range))):
        ft, ff = fl(st), fl(sf)
        by = lambda f, m: {"n": int(m.sum()), "flagged": float(f["any"][m].mean()),
                           **{f"by_{c}": float(f[c][m].mean()) for c in f if c != "any"}}
        det = {fname: by(ff, kind == code) for fname, code in FAULTS.items() if code != 0 and (kind == code).any()}
        if (~dom).any():
            det["contact_asymmetry_gt_max"] = by(ft, ~dom)
        res[ver] = {"detection": det, "false_alarm_test_in_domain": float(ft["any"][dom].mean()),
                    "false_alarm_by_check": {c: float(ft[c][dom].mean()) for c in ft if c != "any"},
                    "faulty_unflagged_k_err": summarize(err_f[~ff["any"]]) if (~ff["any"]).any() else None}
    res["v1"]["thresholds"] = thr1
    res["v2"]["thresholds"] = thr2
    res["v2"]["range_physical"] = {t: [float(np.exp(lo)), float(np.exp(hi))] for t, (lo, hi) in thr2["range"].items()}
    # Phase 6 keys kept for continuity
    summary["faulty"] = {"thresholds_from_val_q99": thr1, "detection": res["v1"]["detection"],
                         "false_alarm_test_in_domain": res["v1"]["false_alarm_test_in_domain"],
                         "faulty_unflagged_k_err": res["v1"]["faulty_unflagged_k_err"]}
    summary["trust_v2"] = res
    (out / "trust_v2.json").write_text(json.dumps(
        {"fitted_on": "val (contact asymmetry <= max_asym)", "thresholds": thr2, "dT_range_K": dT_range,
         "variance_scale": summary.get("variance_scale", {}).get("scale"), "config": tcfg},
        indent=2, default=float), encoding="utf-8")

    names = list(res["v2"]["detection"])
    x = np.arange(len(names) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for j, ver in enumerate(("v1", "v2")):
        vals = [100 * res[ver]["detection"][n]["flagged"] for n in names] + [100 * res[ver]["false_alarm_test_in_domain"]]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=SERIES[j], label=f"trust {ver}")
    ax.set_xticks(x, [FAULT_LABELS.get(n, n) for n in names] + ["false alarm (normal)"], fontsize=8)
    ax.axhline(90, color=INK2, ls="--", lw=1.0, label="plan target 90%")
    ax.set_ylabel("flagged [%]"); ax.set_ylim(0, 105); ax.legend(frameon=False)
    style(ax, "Flagging faulty measurements (thresholds from val" + ("; test second use)" if second_use else ")"))
    save(fig, out / "faulty_detection.png")


def pct(x: float, nd: int = 1) -> str:
    """Percent string: 0.123 -> '12.3%'."""
    return f"{100 * x:.{nd}f}%"


def trust_lines(s: dict) -> list[str]:
    """Markdown lines: trust v1 vs v2 detection table and the variance scale."""
    lines = []
    if "trust_v2" in s:
        r = s["trust_v2"]
        thr = r["v2"]["thresholds"]
        lines += ["| Case | n | v1 | v2 |", "|---|---|---|---|"]
        for n, d in r["v2"]["detection"].items():
            lines.append(f"| {FAULT_LABELS.get(n, n)} | {d['n']} | {pct(r['v1']['detection'][n]['flagged'])} | "
                         f"{pct(d['flagged'])} |")
        lines += [f"| Normal measurement (false alarm) | | {pct(r['v1']['false_alarm_test_in_domain'])} | "
                  f"{pct(r['v2']['false_alarm_test_in_domain'])} |", "",
                  f"v2 thresholds: shared tail share {thr['tail']:.4f}, total false alarm on val "
                  f"{pct(thr['val_false_alarm'])} (budget {pct(thr['budget'], 0)}). "
                  "Range check (prediction range seen on val, SI): "
                  + ", ".join(f"{t[3:]} {lo:.3g}–{hi:.3g}" for t, (lo, hi) in r["v2"]["range_physical"].items()) + "."]
        u = r["v2"]["faulty_unflagged_k_err"]
        if u:
            lines.append(f"Faulty measurements not flagged by v2: median k error {pct(u['med_pct'] / 100)} (n={u['n']}).")
    if "variance_scale" in s:
        v = s["variance_scale"]
        lines += ["", "Variance scale (from val; the predicted std is multiplied by this factor): "
                  + ", ".join(f"{t} × {x:.3f}" for t, x in v["scale"].items()) + ".", "",
                  "| Target | test 68% raw | test 68% scaled | test 95% raw | test 95% scaled |", "|---|---|---|---|---|"]
        for t in v["scale"]:
            a, b = s["calibration_test"][t], v["test_scaled"][t]
            lines.append(f"| {t} | {pct(a['cov68'])} | {pct(b['cov68'])} | {pct(a['cov95'])} | {pct(b['cov95'])} |")
    return lines


def write_trust_report(s: dict, out: Path) -> None:
    """Short markdown report of a --trust-only run."""
    lines = ["# Trust checks v2 and variance scale — M2 v1", "",
             "> **These results are based on simulation only.**", f"> **{s['test_use']}**", "",
             f"Created: {s['created']}. Ensemble: {', '.join(s['ensemble'])}. The model was not retrained.", "",
             *trust_lines(s)]
    (out / "TRUST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_model_card(s: dict, cfg: dict, out: Path) -> None:
    """Model card (markdown) from the evaluation summary."""
    acc = pd.DataFrame(s["test_accuracy"])
    m2 = acc[acc.model == "M2 ensemble"]
    m0 = acc[acc.model == "M0"]
    lines = [
        "# M2 v1 — model card",
        "",
        "> **These results are based on simulation only; validation with real device data has not been done yet.**",
        "",
        f"Created: {s['created']}. Ensemble: {', '.join(s['ensemble'])} ({s['n_params_member']:,} parameters per member).",
        "",
        "## What it does",
        "Estimates ln k, ln α and ln G_c (total sensor–sample interface conductance) from a TPS (hot-disk) "
        "measurement curve, with an uncertainty for each (5-member ensemble: per-model uncertainty + spread between members).",
        "Input: the output of `tpslib.data.features.make_features` (256 log-time bins × 3 channels + 7 scalars). "
        "A raw device record must first go through `tpslib.measurement.preprocess`.",
        "",
        "## Training data",
        "Synthetic data v1 (`data/synthetic/v1`; seeds and config hashes in the manifest): r–z finite-volume simulator, "
        "20-ring copper spiral, Kapton/adhesive layers, separate contact on each face, constant current. "
        f"Training: ~37,600 samples with contact asymmetry below {s['max_contact_asymmetry']:.0f}×. Measurement effects "
        "(noise, drift, t_c, power and calibration errors) were re-drawn every epoch.",
        "",
        "## Valid range",
        "k 0.05–30 W/m·K, ρc_p 0.8–4.5 MJ/m³·K (α ≈ 0.011–37 mm²/s), sample thickness 8–25 mm, radius 20–30 mm, "
        "h_c 500–50,000 W/m²·K on each face with at most a 10× difference, power 8 mW–1 W (auto-setup rule).",
        "",
        "## Test results (locked test set, 5,000 samples, realistic contact domain)",
        "",
        "| k band [W/m·K] | n | M2 k median | M2 k p95 | M0 k median | M0 k p95 | M2 α median | M2 α p95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (_, a), (_, b) in zip(m2.iterrows(), m0.iterrows()):
        lines.append(f"| {a.k_band} | {a.n} | {a.k_med_pct:.2f}% | {a.k_p95_pct:.2f}% | {b.k_med_pct:.2f}% | "
                     f"{b.k_p95_pct:.2f}% | {a.a_med_pct:.2f}% | {a.a_p95_pct:.2f}% |")
    cal = s["calibration_test"]["ln_k"]
    lines += ["", f"Target (simulation): k median < {TARGET_MED:.0f}%, p95 < {TARGET_P95:.0f}%. "
              f"Bands that meet it: {', '.join(b for b, ok in s['test_targets_met'].items() if ok) or 'none'}.",
              f"Uncertainty calibration (ln k): 68% interval coverage {cal['cov68']:.3f}, 95% interval {cal['cov95']:.3f}.",
              "Calibration errors (α_R σ 1%, radius σ 0.5%) and the ±0.5% power error cannot be seen in the curve; "
              "the irreducible floor for k is about 0.77% median, 2.3% p95."]
    if "interpolation" in s:
        it = s["interpolation"]
        lines += ["", "## Interpolation (ablation)",
                  f"A model trained without the k = 1.5–2.5 W/m·K band reaches k median {it['ablation_on_band']['med_pct']:.2f}%, "
                  f"p95 {it['ablation_on_band']['p95_pct']:.2f}% on that band (model that saw the band: "
                  f"{it['main_on_band']['med_pct']:.2f}% / {it['main_on_band']['p95_pct']:.2f}%)."]
    if "trust_v2" in s or "variance_scale" in s:
        lines += ["", "## Faulty measurements and uncertainty scale (thresholds and scale from val)", *trust_lines(s)]
    if "compact" in s:
        c = s["compact"]
        tot = [r for r in c["accuracy"] if r["k_band"] == "all"][0]
        lines += ["", "## Compact device model",
                  f"{c['n_params']:,} parameters ({len(c['runs'])} members): k median {tot['k_med_pct']:.2f}%, "
                  f"p95 {tot['k_p95_pct']:.2f}%; difference to the large ensemble median {c['vs_ensemble_k_med_pct']:.2f}%, "
                  f"p99 {c['vs_ensemble_k_p99_pct']:.2f}%."]
    lines += ["", "## Known limitations",
              "- **Simulation only.** Noise, drift, t_c and contact distribution of the real device are modelled with placeholder values (`configs/device.yaml`).",
              "- **High k (> 5 W/m·K) is the hardest region.** Power clips at 1 W and most of the signal comes from the sensor's own layers; the error is clearly larger.",
              "- **Contact asymmetry > 10 is outside the training domain.** k is underestimated; the trust checks should flag these measurements.",
              "- **Physics not modelled:** lead heating, PETG plate, anisotropy, temperature-dependent properties, sample roughness.",
              "- **Single sensor geometry.** a = 6.4 mm, 20 rings (variant A). Another geometry needs new data and training.",
              "", "## Files",
              "`onnx/m2_ensemble.onnx` (normalisation built in; input: raw features, output: mean and std in ln units); "
              f"largest difference to PyTorch: {s['onnx']['max_abs_diff_vs_pytorch']}."]
    (out / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
