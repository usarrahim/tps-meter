# Faz 0 raporu — İskelet ve ortam (2026-10-17)

## Yapılanlar

- İstenen klasör yapısı kuruldu (kök: `tps-master/`). Henüz yazılmamış modüller, hangi fazda dolacağını söyleyen taslak dosyalar.
- `pyproject.toml` (numpy, scipy, h5py, pyyaml, torch, matplotlib, pandas, tqdm, lightgbm, onnx; geliştirme için pytest), `.gitignore` (`data/`, `runs/`, `.venv/`, `*.h5`, `*.pt`, `*.onnx`).
- Config dosyaları: `sensor.yaml` PROJECT_RULES.md değerleriyle; `prior_v1.yaml`, `dataset_v1.yaml`, `train_m2.yaml` prompttaki değerlerle; `device.yaml` ve `sim_numerics.yaml` başlangıç değerleriyle (yer tutucular işaretli).
- `tpslib.runtime.get_device()`: CUDA varsa onu, yoksa CPU'yu seçer. `TPS_DEVICE` ile zorlanabilir.
- `tpslib.config`: YAML → dataclass; mm/m karışıklığını yakalayan birim aralığı kontrolü.
- Colab altyapısı: `notebooks/colab_run.ipynb`, `scripts/pack_for_colab.py`, `docs/COLAB.md`.
- Ortam: Python 3.11.9 (`py install` ile kuruldu), `.venv`, CUDA'lı PyTorch. Ayrıntılar aşağıda.

## Ortam

| | |
|---|---|
| Python | 3.11.9 (`.venv`) |
| PyTorch | 2.11.0+cu128, CUDA 12.8 |
| `get_device()` | `cuda:0 NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.0 GB)` |
| CPU / RAM | i5-12500H, 16 iş parçacığı / 15,7 GB |
| Diğer | numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pandas 3.0.6, matplotlib 3.11.2, lightgbm 4.7.0, onnx 1.23.2, pytest 9.1.1 |

Not: İnternet bağlantısı torch indirmesi sırasında 6 kez koptu; wheel önce `pip download --resume-retries` ile indirildi, sonra yerelden kuruldu. Yeniden kurulumda aynı sorun çıkarsa bu yol kullanılabilir.

## Kapı

`python -m pytest -q` → **4 passed** (config'ler yükleniyor, sensör değerleri PROJECT_RULES.md ile aynı, birim koruması mm'yi reddediyor, `get_device()` CUDA'da tensör oluşturuyor). ✔
`scripts/pack_for_colab.py` → `dist/tps-meter-code.zip` (46 dosya, 21 KB) üretildi. ✔ (Colab defteri henüz gerçek Colab'da çalıştırılmadı.)

Bu fazda grafik yok (çizilecek bir sonuç yok).

## Fazların özeti (kendi cümlelerimle)

1. **Faz 1 — Analitik çözüm ve M0.** Gustafsson D(τ)'yu m halkalı olarak yazıp PROJECT_RULES.md'deki 6 değere karşı test ediyorum; hızlı kullanım için tablo + interpolasyon. M0: α'yı tarayıp ΔT'nin D(τ)'ya en doğrusal olduğu α'yı bulan, eğimden k'yi çıkaran klasik uydurma. Kapı: gürültüsüz analitik eğrilerde 20 farklı k için k ve α hatası < %0,5.
2. **Faz 2 — Simülatör.** r–z sonlu hacim, iki numune + aralarında ince katmanlar (toplu direnç + kapasite), iki yüzde ayrı h_c, dış yüzeyde zayıf taşınım, logaritmik zaman adımı, seyrek LU. Kaynak: uzunlukla orantılı 20 halka, sabit akım geri beslemesi, uzunluk ağırlıklı okuma. Kapılar: analitik sınır %0,5, yakınsama %0,1, enerji %0,5, fiziksel davranış grafikleri, < 1 s/simülasyon. **Sonunda kesin durak.**
3. **Faz 3 — Prior ve veri.** Parametreleri bağımsız örnekleyip, güç/süre/veri hızını cihaz firmware'iyle aynı otomatik ayar kuralından türetiyorum. Paralel, devam ettirilebilir HDF5 üretimi + manifest. Önce 1.000 örneklik pilot ve histogramlar, onaydan sonra 5 bölme.
4. **Faz 4 — Ölçüm etkileri ve özellikler.** Temiz eğriye eğitim anında gürültü/kayma/ADC/güç hatası eklenir; gerçek ve sentetik veri aynı `preprocess` + `features` yolundan geçer. 256 noktalı log-zaman ızgarası, θ ve dθ/d ln t kanalları + skalerler. Normalizasyon yalnızca train'den.
5. **Faz 5 — M2.** 1B ResNet, β-NLL kaybı, 5 tohumlu topluluk + k bandı çıkarılmış ablasyon modeli; önce 5 epoklu deneme.
6. **Faz 6 — Değerlendirme.** k aralıklarına göre hata, kalibrasyon, M0–M2 karşılaştırması, interpolasyon ve hatalı ölçüm testleri, ONNX dışa aktarma, kompakt (~60 bin) model, model kartı.

## Açık sorular / çelişkiler

1. **h_c çıktısı.** Prior'da h_c alt ve üst ayrı örnekleniyor ama model tek `log h_c` çıkarıyor. Önerim: hedef olarak iki tarafın toplamı `h_eff = h_alt + h_üst` (sensörden iki numuneye giden yollar paralel) olsun. Uygun mu, yoksa iki ayrı çıktı mı?
2. **Parametre sayısı.** Blok başına iki konvolüsyon (standart ResNet) ile 32–64–128–128, çekirdek 5 yaklaşık **330 bin** parametre ediyor; ~200 bin hedefiyle çelişiyor. Önerim: kanal yapısını koruyup son iki blokta çekirdeği 3 yapmak (~210 bin). Öncelik hangisi: mimari tarif mi, 200 bin mi?
3. **test_faulty ve "temiz veri" kuralı.** Simülatör temiz eğri saklıyor; ama "kayma sıçraması" bir ölçüm etkisi. Önerim: test_faulty dosyasına temiz eğri + arıza etiketi yazılır, sıçrama değerlendirmede sabit tohumla aynı `acquisition` yolundan eklenir. Uygun mu?
4. **Enerji korunumu kapısı.** Dış yüzeylerde taşınım kaybı var; denge "verilen = depolanan + dış yüzeylerden kaybolan" olarak test edilecek. Doğru anlıyor muyum?
5. **M0 penceresi.** Promptta 0,3 < τ² < 1 yazıyor; literatürde yaygın olan 0,33 < τ < 1. τ² mi kastedildi (yani yaklaşık 0,55 < τ < 1), yoksa τ mu?
6. **Hız kapısı.** < 1 s/simülasyon ile %0,1 yakınsama aynı ağda zor olabilir. Çatışırsa önceliğim doğruluk; hız için üretimde paralellik (16 iş parçacığı) kullanırım. 60.000 simülasyon × ~1 s / 14 süreç ≈ 1–1,5 saat, yani üretim yerelde makul görünüyor.
7. **Colab işbölümü.** Önerim: simülasyon ve veri üretimi yerelde (Colab Pro'nun CPU sayısı az), M2 eğitimi Colab GPU'da. Yerel RTX 3050 Ti (4 GB) bu boyutta bir model için de yeterli olabilir; Faz 5'teki 5 epoklu denemede ikisini ölçüp karar verelim. Kodu Drive'a **zip paketiyle** mi, yoksa **Google Drive masaüstü eşitlemesiyle** mi taşımak istersin?
