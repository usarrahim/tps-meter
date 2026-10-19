# tps-meter — Yapay zekâ destekli hot-disk (TPS) ısıl iletkenlik ölçümü

İTÜ EHB bitirme projesi: düşük maliyetli, USB ile çalışan, yapay zekâ destekli bir **hot-disk (Transient Plane Source)** ısıl iletkenlik ölçüm cihazı.
Bu depo cihazın **analiz yazılımını** içerir:
- sensör + numune sisteminin fizik simülatörü,
- sentetik veri üretim hattı,
- klasik TPS çözümü (M0),
- ısıl iletkenlik (k), ısıl yayınım (α) ve temas iletkenliğini belirsizlikleriyle tahmin eden sinir ağı (M2).

> **Gizlilik.** Bu depo tescilli yazılımdır; bkz. [LICENSE](LICENSE). Eğitilmiş model ağırlıkları, ONNX dosyaları, normalizasyon istatistikleri, üretilen veri ve ölçüm verileri **depoya dahil değildir**. Bunlar ayrı, özel bir depolamada tutulur.

## Durum

| Aşama | Durum | Rapor |
|---|---|---|
| Faz 0–1: altyapı, analitik D(τ), klasik TPS (M0) | tamamlandı | [phase0](reports/phase0.md), [phase1](reports/phase1.md) |
| Faz 2: r–z sonlu hacim simülatörü ve doğrulama | tamamlandı | [phase2](reports/phase2.md) |
| Faz 3: prior, otomatik ayar kuralı, 64.000 sentetik eğri | tamamlandı | [phase3](reports/phase3.md) |
| Faz 4: ölçüm zinciri, ortak ön işleme, özellikler | tamamlandı | [phase4](reports/phase4.md) |
| Faz 5–6: M2 topluluğu, test değerlendirmesi, ONNX | tamamlandı (base model M2 v1) | [phase5](reports/phase5.md), [phase6](reports/phase6.md), [model kartı](models/m2_v1/MODEL_CARD.md) |
| Gerçek cihaz verisiyle kalibrasyon ve doğrulama | sırada | — |

**M2 v1 test sonucu** (yalnızca simülasyon; k medyan / %95'lik hata):

| k aralığı | M2 v1 | M0 (klasik) |
|---|---|---|
| 0,05–0,5 W/m·K | %0,83 / %2,6 | %1,21 / %3,7 |
| 0,5–5 W/m·K | %0,89 / %3,2 | %1,41 / %6,4 |
| 5–30 W/m·K | %2,07 / %10,4 | %5,03 / %23,9 |

Gerçek cihaz doğruluğu henüz ölçülmedi.

## Depo düzeni

```
configs/        tüm ayarlar (SI birimleri): sensör, cihaz, simülasyon sayısalları, prior, veri seti, eğitim
src/tpslib/
  dtau.py, fit_tps.py        analitik D(τ) ve klasik TPS uydurması (M0)
  sim/                       r–z sonlu hacim simülatörü (ağ, çözücü, sensör modeli, analitik referans)
  measurement/               ölçüm zinciri modeli ve gerçek/sentetik için ORTAK ön işleme
  data/                      prior + otomatik ayar, paralel veri üretimi, özellikler, PyTorch dataset
  models/                    1B ResNet (M2), β-NLL kaybı, ONNX dışa aktarma
  train/                     eğitim döngüsü (devam ettirilebilir)
  eval/                      metrikler, tahmin, M0 toplu çalıştırma, güven kontrolleri, grafik stili
scripts/        komut satırı girişleri (doğrulama, üretim, eğitim, teşhis, değerlendirme)
tests/          pytest (fizik fonksiyonlarının her birinin testi var)
notebooks/      Google Colab çalıştırıcısı
reports/        faz raporları ve grafikler
docs/           ilerleme, tasarım kararları, Colab kılavuzu, proje özeti
models/         yalnızca model kartları (ağırlıklar depoda değil)
```

## Kurulum (Windows, PowerShell)

Python 3.11 gerekir.

```powershell
py -V:3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Ortamı etkinleştirme ve testler:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## Sık kullanılan komutlar

| İş | Komut |
|---|---|
| Simülatör doğrulama kapıları | `python scripts/validate_sim.py` |
| Veri üretimi (devam ettirilebilir) | `python scripts/generate_dataset.py --splits train val test test_interp test_faulty` |
| M2 eğitimi (5 tohum) | `python scripts/train_m2.py --seeds 1 2 3 4 5 --tag v1sym --max-asym 10` |
| Hata teşhisi (val) | `python scripts/diagnose.py --run v1sym_m2_seed1` |
| Son değerlendirme + ONNX | `python scripts/evaluate.py --ensemble-tag v1sym ...` |

Uzun işler Google Colab'da çalıştırılır: [docs/COLAB.md](docs/COLAB.md), [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb). Veri ve eğitim çıktıları için `TPS_DATA_ROOT` ve `TPS_RUNS_ROOT` ortam değişkenleri kullanılır.

## Belgeler

- [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): faz faz iş tanımı
- [docs/PROGRESS.md](docs/PROGRESS.md): ilerleme günlüğü
- [docs/DECISIONS.md](docs/DECISIONS.md): tasarım kararları ve gerekçeleri
- [docs/PROJECT_RULES.md](docs/PROJECT_RULES.md): proje kuralları ve fizik sabitleri (tek doğruluk kaynağı: `configs/sensor.yaml`)

## Lisans

Tescilli; tüm hakları saklıdır. Bkz. [LICENSE](LICENSE).
