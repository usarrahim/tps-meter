"""Phase 6: final evaluation on the locked test splits, M0 comparison, calibration, interpolation
(ablation), faulty-measurement detection, compact model, ONNX export and the model card.

The test splits are used ONCE here; all thresholds come from the validation split.

Usage (Colab):
  python scripts/evaluate.py --ensemble-tag v1sym --seeds 1 2 3 4 5 --ablation ablation_m2_seed1 \
      --compact-tag v1c --compact-seeds 1 --max-asym 10
Writes <runs root>/../reports/phase6/ (figures, CSV, summary.json, MODEL_CARD.md, onnx/) and
<runs root>/../phase6_bundle.zip with all of it.
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
from tpslib.eval.trust import fit_thresholds, flags, trust_scores
from tpslib.models.export import EnsembleWithNorm, export_onnx, verify_onnx
from tpslib.models.resnet1d import count_parameters
from tpslib.runtime import runs_root

TARGET_MED, TARGET_P95 = 1.0, 4.0
SPLITS = ("val", "test", "test_interp", "test_faulty")


def band_table(err: dict, k: np.ndarray, sel: np.ndarray, label: str) -> list[dict]:
    rows = []
    for lo, hi in [*K_BANDS, (0.05, 30.01)]:
        m = sel & (k >= lo) & (k < hi)
        ek, ea = summarize(err["k"][m]), summarize(err["alpha"][m])
        rows.append({"model": label, "k_band": f"{lo}-{min(hi, 30)}" if hi < 30.01 else "tümü", "n": ek["n"],
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
    ap.add_argument("--out", default=None, help="default: <runs root>/../reports/phase6")
    args = ap.parse_args()
    t_start = time.perf_counter()

    root = Path(runs_root())
    out = Path(args.out) if args.out else root.parent / "reports" / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "onnx").mkdir(exist_ok=True)
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
    for split in SPLITS:
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
               "targets": {"k_med_pct": TARGET_MED, "k_p95_pct": TARGET_P95}}

    # --- test: accuracy per k band, M2 vs M0 -----------------------------------------------------------
    p, e = params["test"], ens["test"]
    dom = in_domain(p)
    err, err0 = rel_errors(e), m0_errors(m0["test"], p)
    rows = band_table(err, p["k"], dom, "M2 topluluk") + band_table(err0, p["k"], dom & m0["test"]["ok"], "M0")
    rows += band_table(err, p["k"], ~dom, "M2 (asimetri > 10, arıza alanı)")
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "test_accuracy_by_band.csv", index=False)
    summary["test_accuracy"] = rows
    summary["test_targets_met"] = {r["k_band"]: bool(r["k_med_pct"] < TARGET_MED and r["k_p95_pct"] < TARGET_P95)
                                   for r in rows if r["model"] == "M2 topluluk"}
    summary["m0_fail_rate_test"] = float(1 - m0["test"]["ok"].mean())

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, (lab, ee, c, mk) in zip(axs[:2], [("M2 topluluk", err, SERIES[0], "o"), ("M0 (klasik)", err0, SERIES[1], "s")]):
        ax.axhspan(-TARGET_P95, TARGET_P95, color=BAND, zorder=0)
        ax.scatter(p["k"][dom], 100 * ee["k"][dom], s=3, color=c, alpha=0.5)
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xscale("log"); ax.set_ylim(-30, 30)
        ax.set_xlabel("k [W/m·K]"); ax.set_ylabel("k bağıl hatası [%]")
        style(ax, f"{lab}, test (gerçekçi temas); gri: ±%{TARGET_P95:.0f}")
    hmin = np.minimum(p["h_c_bottom"], p["h_c_top"])
    axs[2].scatter(hmin[dom], 100 * err["k"][dom], s=3, color=SERIES[0], alpha=0.5)
    axs[2].set_xscale("log"); axs[2].set_ylim(-30, 30); axs[2].axhline(0, color=INK2, lw=0.8)
    axs[2].set_xlabel("min(h_c alt, h_c üst) [W/m²·K]"); axs[2].set_ylabel("k bağıl hatası [%]")
    style(axs[2], "M2 hatası – temas iletkenliği")
    save(fig, out / "test_error_scatter.png")

    # --- calibration ------------------------------------------------------------------------------------
    levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95, 0.99])
    from scipy.stats import norm
    cal = {}
    for i, t in enumerate(("ln_k", "ln_alpha", "ln_G_c")):
        z = np.abs(e["mean"][dom, i] - e["true"][dom, i]) / e["std"][dom, i]
        cal[t] = [float(np.mean(z <= norm.ppf(0.5 + lv / 2))) for lv in levels]
    summary["calibration_test"] = {t: {"cov68": cal[t][6], "cov95": cal[t][10]} for t in cal}
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="ideal")
    for i, t in enumerate(cal):
        ax.plot(levels, cal[t], MARKERS[i] + "-", color=SERIES[i], ms=5, label=t)
    ax.set_xlabel("beklenen kapsama"); ax.set_ylabel("gerçekleşen kapsama"); ax.legend(frameon=False)
    style(ax, "Belirsizlik kalibrasyonu (test, gerçekçi temas)")
    save(fig, out / "calibration.png")

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
        ax.scatter(pi["k"][di], 100 * ei["k"][di], s=4, color=SERIES[0], alpha=0.5, label="ana model (bandı gördü)")
        ax.scatter(pi["k"][di], 100 * ea["k"][di], s=4, color=SERIES[1], alpha=0.5, label="ablasyon (bandı görmedi)")
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xlabel("k [W/m·K]"); ax.set_ylabel("k hatası [%]")
        ax.legend(frameon=False); style(ax, "test_interp: k = 1,5–2,5 W/m·K bandı")
        save(fig, out / "interpolation.png")

    # --- faulty measurements / trust checks --------------------------------------------------------
    if "test_faulty" in ens:
        dv = in_domain(params["val"])
        sub = lambda d, m: {k: v[m] for k, v in d.items()}
        sv = trust_scores(sub(ens["val"], dv), sub(m0["val"], dv))
        thr = fit_thresholds(sv, q=0.99)
        ft = flags(trust_scores(ens["test"], m0["test"]), thr)
        ff = flags(trust_scores(ens["test_faulty"], m0["test_faulty"]), thr)
        kind = params["test_faulty"]["fault"]
        det = {}
        for name, code in FAULTS.items():
            if code == 0:
                continue
            m = kind == code
            if m.any():
                det[name] = {"n": int(m.sum()), "flagged": float(ff["any"][m].mean()),
                             **{f"by_{c}": float(ff[c][m].mean()) for c in ff if c != "any"}}
        err_f = rel_errors(ens["test_faulty"])["k"]
        unflag = ~ff["any"]
        summary["faulty"] = {
            "thresholds_from_val_q99": thr, "detection": det,
            "false_alarm_test_in_domain": float(ft["any"][dom].mean()),
            "flagged_test_out_of_domain_asym": float(ft["any"][~dom].mean()) if (~dom).any() else None,
            "faulty_unflagged_k_err": summarize(err_f[unflag]) if unflag.any() else None,
            "faulty_flagged_k_err": summarize(err_f[ff["any"]]) if ff["any"].any() else None}
        fig, ax = plt.subplots(figsize=(8, 4.2))
        names = list(det)
        ax.bar([n.replace("_", " ") for n in names], [100 * det[n]["flagged"] for n in names], color=SERIES[0])
        ax.axhline(90, color=SERIES[1], ls="--", lw=1.2, label="plan hedefi %90")
        ax.axhline(100 * summary["faulty"]["false_alarm_test_in_domain"], color=INK2, ls=":", lw=1.2,
                   label="normal ölçümde yanlış alarm")
        ax.set_ylabel("işaretlenen [%]"); ax.set_ylim(0, 105); ax.legend(frameon=False)
        style(ax, "Hatalı ölçümlerin işaretlenmesi (eşikler val'dan)")
        save(fig, out / "faulty_detection.png")

    # --- compact model ----------------------------------------------------------------------------------
    comp_runs = [root / f"{args.compact_tag}_m2c_seed{s}" for s in args.compact_seeds]
    comp_runs = [r for r in comp_runs if (r / "best.pt").exists()]
    if comp_runs:
        cl = [load_run(r) for r in comp_runs]
        ce = ensemble(predict_members([(m, nm) for m, nm, _ in cl], ds["test"], num_workers=nw))
        crow = band_table(rel_errors(ce), p["k"], dom, f"kompakt ({len(cl)} üye)")
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

    summary["elapsed_s"] = time.perf_counter() - t_start
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    write_model_card(summary, cfg, out)
    bundle = shutil.make_archive(str(out.parent.parent / "phase6_bundle"), "zip", out)
    log(f"done -> {out}  bundle: {bundle}")
    print(acc.round(2).to_string(index=False))
    print(json.dumps({k: summary[k] for k in summary if k not in ("test_accuracy",)}, indent=1,
                     ensure_ascii=False, default=float)[:6000])


def write_model_card(s: dict, cfg: dict, out: Path) -> None:
    acc = pd.DataFrame(s["test_accuracy"])
    m2 = acc[acc.model == "M2 topluluk"]
    m0 = acc[acc.model == "M0"]
    lines = [
        "# M2 v1 — model kartı",
        "",
        "> **Bu sonuçlar yalnızca simülasyona dayanır; gerçek cihaz verisiyle doğrulama henüz yapılmadı.**",
        "",
        f"Oluşturma: {s['created']}. Topluluk: {', '.join(s['ensemble'])} (üye başına {s['n_params_member']:,} parametre).",
        "",
        "## Ne yapar",
        "TPS (hot-disk) ölçüm eğrisinden ln k, ln α ve ln G_c (sensör–numune toplam arayüz iletkenliği) tahmini ve "
        "her biri için belirsizlik (5 üyeli topluluk: model belirsizliği + üyeler arası fark).",
        "Girdi: `tpslib.data.features.make_features` çıktısı (256 log-zaman bini × 3 kanal + 7 skaler). "
        "Ham cihaz kaydı önce `tpslib.measurement.preprocess` ile işlenmelidir.",
        "",
        "## Eğitim verisi",
        "Sentetik veri v1 (`data/synthetic/v1`, manifest'te tohumlar ve config özetleri): r–z sonlu hacim simülatörü, "
        "20 halkalı bakır spiral, Kapton/yapıştırıcı katmanları, iki yüzde ayrı temas, sabit akım. "
        f"Eğitim: {s['max_contact_asymmetry']:.0f} kattan küçük temas asimetrili ~37.600 örnek. Ölçüm etkileri (gürültü, kayma, "
        "t_c, güç ve kalibrasyon hataları) her epokta yeniden eklendi.",
        "",
        "## Geçerli aralık",
        "k 0,05–30 W/m·K, ρc_p 0,8–4,5 MJ/m³·K (α ≈ 0,011–37 mm²/s), numune kalınlığı 8–25 mm, yarıçapı 20–30 mm, "
        "iki yüzde h_c 500–50.000 W/m²·K ve aralarında en fazla 10 kat fark, güç 8 mW–1 W (otomatik ayar kuralıyla).",
        "",
        "## Test sonuçları (kilitli test seti, 5.000 örnek, gerçekçi temas alanı)",
        "",
        "| k bandı [W/m·K] | n | M2 k medyan | M2 k %95 | M0 k medyan | M0 k %95 | M2 α medyan | M2 α %95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (_, a), (_, b) in zip(m2.iterrows(), m0.iterrows()):
        lines.append(f"| {a.k_band} | {a.n} | %{a.k_med_pct:.2f} | %{a.k_p95_pct:.2f} | %{b.k_med_pct:.2f} | "
                     f"%{b.k_p95_pct:.2f} | %{a.a_med_pct:.2f} | %{a.a_p95_pct:.2f} |")
    cal = s["calibration_test"]["ln_k"]
    lines += ["", f"Hedef (simülasyon): k medyan < %{TARGET_MED:.0f}, %95'lik < %{TARGET_P95:.0f}. "
              f"Karşılanan bantlar: {', '.join(b for b, ok in s['test_targets_met'].items() if ok) or 'yok'}.",
              f"Belirsizlik kalibrasyonu (ln k): %68 aralık kapsaması {cal['cov68']:.3f}, %95 aralık {cal['cov95']:.3f}.",
              "Kalibrasyon hataları (α_R σ %1, yarıçap σ %0,5) ve ±%0,5 güç hatası eğriden görülemez; "
              "k için kaldırılamaz taban yaklaşık %0,77 medyan, %2,3 %95'lik."]
    if "interpolation" in s:
        it = s["interpolation"]
        lines += ["", "## İnterpolasyon (ablasyon)",
                  f"k = 1,5–2,5 W/m·K bandı eğitimden çıkarılan model bu bantta k medyan %{it['ablation_on_band']['med_pct']:.2f}, "
                  f"%95 %{it['ablation_on_band']['p95_pct']:.2f} (bandı gören model: %{it['main_on_band']['med_pct']:.2f} / "
                  f"%{it['main_on_band']['p95_pct']:.2f})."]
    if "faulty" in s:
        f = s["faulty"]
        lines += ["", "## Hatalı ölçümler (güven kontrolleri, eşikler val'dan %99'luk)",
                  *[f"- {k.replace('_', ' ')}: %{100 * v['flagged']:.1f} işaretlendi (n={v['n']})" for k, v in f["detection"].items()],
                  f"- Normal test ölçümlerinde yanlış alarm: %{100 * f['false_alarm_test_in_domain']:.1f}"]
    if "compact" in s:
        c = s["compact"]
        tot = [r for r in c["accuracy"] if r["k_band"] == "tümü"][0]
        lines += ["", "## Kompakt cihaz modeli",
                  f"{c['n_params']:,} parametre ({len(c['runs'])} üye): k medyan %{tot['k_med_pct']:.2f}, %95 %{tot['k_p95_pct']:.2f}; "
                  f"büyük toplulukla fark medyan %{c['vs_ensemble_k_med_pct']:.2f}, %99'luk %{c['vs_ensemble_k_p99_pct']:.2f}."]
    lines += ["", "## Bilinen sınırlar",
              "- **Yalnızca simülasyon.** Gerçek cihazın gürültüsü, kayması, t_c'si ve temas dağılımı yer tutucu değerlerle modellendi (`configs/device.yaml`).",
              "- **Yüksek k (> 5 W/m·K) en zor bölge.** Güç 1 W'ta kırpılıyor, sinyalin çoğu sensörün kendi katmanlarından geliyor; hata belirgin büyük.",
              "- **Temas asimetrisi > 10 eğitim alanı dışında.** k düşük tahmin edilir; güven kontrolleri bu ölçümleri işaretlemeli.",
              "- **Modellenmeyen fizik:** uç ısınması, PETG plaka, anizotropi, sıcaklığa bağlı özellikler, numune pürüzlülüğü.",
              "- **Tek sensör geometrisi.** a = 6,4 mm, 20 halka (varyant A). Başka geometri için yeniden veri ve eğitim gerekir.",
              "", "## Dosyalar",
              f"`onnx/m2_ensemble.onnx` (normalizasyon içinde; girdi: ham özellikler, çıktı: ln birimlerinde ortalama ve std); "
              f"PyTorch ile en büyük fark: {s['onnx']['max_abs_diff_vs_pytorch']}."]
    (out / "MODEL_CARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
