# İlerleme

| Faz | Durum | Kapı | Rapor |
|---|---|---|---|
| 0 — İskelet ve ortam | Onaylandı | `pytest` hatasız ✔ | [reports/phase0.md](../reports/phase0.md) |
| 1 — Analitik çözüm ve M0 | Onaylandı | D(τ) %0,0007; M0 k ≤ %0,003, α ≤ %0,019 (20 k, sabit akım dahil) ✔ | [reports/phase1.md](../reports/phase1.md) |
| 2 — Simülatör ve doğrulama | Onaylandı (Faz 3'te şema TR-BDF2'ye geçti, kapılar yeniden geçti) | Analitik %0,26 ✔, yakınsama %0,099 ✔, enerji ✔, fizik ✔, hız 2,1–2,6 s ✘ (bilinçli) | [reports/phase2.md](../reports/phase2.md) |
| 3 — Prior ve veri üretimi | Tamamlandı (Colab) | 64.000/64.000, 0 hata ✔ | [reports/phase3.md](../reports/phase3.md) |
| 4 — Ölçüm etkileri ve özellikler | Onaylandı | M0 + ölçüm etkileri k ≤ %0,70 ✔; normalizasyon ✔ | [reports/phase4.md](../reports/phase4.md) |
| 5 — M2 eğitimi | Tamamlandı (Colab): 5 üyeli `v1sym` + ablasyon + kompakt | Val k medyan %1,05–1,12, %95'lik %6,1–6,4 ✔ | [reports/phase5.md](../reports/phase5.md) |
| 6 — Değerlendirme ve dışa aktarma | Tamamlandı, **onay bekliyor** | k<5: hedef ✔; 5–30: ✘ (%95'lik %10); ONNX ✔; arıza tespiti kısmen | [reports/phase6.md](../reports/phase6.md) |

## Faz 0 — 2026-10-17

- Klasör yapısı, `pyproject.toml`, `.gitignore`, 6 config dosyası, modül taslakları kuruldu; git deposu başlatıldı (henüz commit yok).
- `.venv`: Python 3.11.9 + CUDA'lı PyTorch; GPU: RTX 3050 Ti Laptop (4 GB).
- `tpslib.runtime.get_device()`: CUDA varsa onu, yoksa CPU'yu seçer.
- `tpslib.config`: YAML yükleyici + `SensorConfig` (SI birim aralığı kontrolüyle).
- Colab: `notebooks/colab_run.ipynb`, `scripts/pack_for_colab.py`, `docs/COLAB.md`.
- Açık sorular kullanıcıya soruldu; kararlar `docs/DECISIONS.md` içinde.

## Faz 1 — 2026-10-17

- `dtau.py`: D(τ) doğrudan hesap + tablo; sabit güç ve sabit akım için ideal ΔT.
- `fit_tps.py`: M0, ΔT = A·D̃ + B·P(t)/P₀, 0,3 < τ² < 1 penceresi, α taraması.
- Bulgular: D(τ) alt sınırı σ_min = 10⁻⁶ (k'yi etkilemez); sabit akım ihmal edilirse k %0,7, α %2 hata.
- Testler: 65 passed. Rapor ve grafikler: `reports/phase1.md`, `reports/phase1/`.

## Faz 2 — 2026-10-17

- Simülatör: `sim/geometry.py` (iz hizalı, kenarda inceltilmiş, iki bölgeli ağ), `sim/solver.py` (sonlu hacim, Richardson'lı kapalı Euler, Sherman–Morrison ile sabit akım), `sim/sensor_model.py` (`simulate()`), `sim/analytic.py` (sonlu iz genişlikli analitik referans).
- `scripts/validate_sim.py`: beş kapı + grafikler. Hız dışındaki kapılar geçti; hız 2,8–3,5 s/sim (doğruluk tercihi).
- Ağ çalışması bulguları: bakır iz kenarında ~20 µm sınır tabakası; α'nın uzak alan büyüme oranlarına ve zaman adımına duyarlılığı.
- Testler: 78 passed. Kullanıcıyla grafik incelemesi bekleniyor (prompt gereği zorunlu durak).

## Faz 3 (pilot) — 2026-10-17

- `data/prior.py` (prior + otomatik ayar + arızalar + G_c), `data/generate.py` ve `scripts/generate_dataset.py` (paralel, devam ettirilebilir, HDF5, manifest), `scripts/pilot_report.py`.
- Pilot: 1.000 örnek, 0 hata, 1,44 örnek/s. Tam üretim tahmini ~12 saat (darboğaz bellek bant genişliği).
- Simülatör zaman şeması TR-BDF2'ye geçti (paralel verim +%50); Faz 2 kapıları yeniden geçti.
- YAML'da `0.8e6` gibi değerlerin metin okunması hatası düzeltildi ve testlendi.
- Testler: 90 passed. Tam üretim için kullanıcı onayı bekleniyor.

## Faz 3 (tam üretim) — 2026-10-17/18

- Kullanıcı tüm üretimi Colab'da yapmayı seçti. Üreticiye Drive yedekleme/geri yükleme (`--mirror`) ve kod özeti koruması eklendi; `scripts/bench_generation.py` ve Colab defteri yazıldı.
- v5e-1 TPU çalışma zamanı (24 CPU): 3,37 örnek/s → ~5,3 saat. Üretim kullanıcının Colab oturumunda sürüyor.

## Faz 4 — 2026-10-18

- `measurement/acquisition.py` (temiz eğri → ham cihaz kaydı, tüm ölçüm etkileri), `measurement/preprocess.py` (gerçek ve sentetik için ortak), `data/features.py` (256 log bin, 3 kanal, 7 skaler), `data/dataset.py` (epok başına yeni etkiler, yalnızca train'den normalizasyon).
- Kapı: M0 ideal eğri + ölçüm etkileri k ≤ %0,70 (kalibrasyonla %1,59). α'yı uzun ve düşük k'lı ölçümlerde ortam kayması belirliyor (%5–10).
- Testler: 125 passed. `device.yaml` değişti: Colab üretimi bitmeden yeni zip yüklenmemeli.

## Faz 5 (kod) — 2026-10-18

- `models/resnet1d.py` (375k parametre), `models/losses.py` (β-NLL), `train/trainer.py` (AMP, kosinüs + ısınma, erken durdurma, devam ettirilebilir koşular, epok kodlayan örnekleyici), `scripts/train_m2.py` (tohumlar, ablasyon, veri miktarı), Colab hücreleri 7–12.
- Pilot denemesi (900/100, 40 epok, yığın 32): k medyan %1,8, α medyan %9, cov95 %96. Kod doğrulandı.
- Testler: 131 passed. Tam eğitim Colab üretimi bitince yapılacak.

## Faz 5 (eğitim) ve Faz 6 (kod) — 2026-10-18

- 10k modeli ve teşhis: kuyruk yüksek k ve aşırı temas asimetrisinden geliyor. Kullanıcı kararı: asimetri > 10 arıza; topluluk `v1sym` (5 tohum, asimetri ≤ 10) Colab'da eğitiliyor. Erken durdurma 30. epoktan önce devreye girmiyor.
- Faz 6 kodu: `scripts/evaluate.py` (k bantları, M0 karşılaştırması, kalibrasyon, interpolasyon/ablasyon, arıza tespiti, kompakt model, ONNX + doğrulama, otomatik model kartı, paket zip), `eval/m0_batch.py`, `eval/trust.py`, `models/export.py`, `configs/train_m2_compact.yaml` (61.910 parametre). Pilot verisiyle kuru çalıştırma tamam; ONNX ile PyTorch farkı 2e-6.
- Testler: 140 passed.

## Faz 6 — 2026-10-19

- Kilitli test seti (gerçekçi temas, n = 3.715):
  - k medyan / %95'lik: 0,05–0,5 → %0,83 / %2,60 · 0,5–5 → %0,89 / %3,17 · 5–30 → %2,07 / %10,36.
  - M0 tümünde %1,71 / %13,47.
- Kalibrasyon: ln k %95 kapsama %97. Ablasyon modeli görmediği bantta %0,92 / %2,91.
- Arıza tespiti: ince numune %98, kayma sıçraması %85, hava boşluğu %53. G_c aralık kontrolü ve yeniden simülasyon (M0+) eksik.
- Kompakt model 2 × 61.910 parametre, toplulukla fark %0,27. ONNX doğrulandı (2e-6).
- Model kartı: `models/m2_v1/MODEL_CARD.md`. ONNX ve ağırlıklar Drive'da (git dışı).
