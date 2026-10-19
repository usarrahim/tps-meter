Bu depoda TPS (hot-disk) ısıl iletkenlik cihazımızın simülatörünü, sentetik veri üretim hattını ve k/α tahmin modelini (M0 ve M2) kuracağız. Önce PROJECT_RULES.md dosyasını oku; oradaki fizik sabitleri ve kurallar bu görevin tamamında geçerli.

İşi aşağıdaki fazlarla yap. Her fazın sonunda: testleri çalıştır, `reports/` altına Türkçe kısa bir özet ve grafikler yaz, `docs/PROGRESS.md` dosyasını güncelle, sonra DUR ve onayımı bekle. Bir doğrulama kapısı geçmezse sonraki faza geçme; nedenini araştır ve bana raporla. Başlamadan önce bu fazları kendi cümlelerinle kısaca özetleyip anlamadığın ya da çelişkili bulduğun nokta varsa sor.

## Klasör yapısı

```
tps-meter/
├── PROJECT_RULES.md
├── README.md
├── pyproject.toml
├── .gitignore                  # data/, runs/, .venv/, *.h5, *.pt
├── configs/
│   ├── sensor.yaml             # sensör ve katman geometrisi (tek doğruluk kaynağı)
│   ├── device.yaml             # güç aralığı, akım aralıkları, veri hızları, gürültü modeli
│   ├── sim_numerics.yaml       # ağ, zaman adımı, çözücü ayarları
│   ├── prior_v1.yaml           # simülasyon parametre aralıkları ve dağılımları
│   ├── dataset_v1.yaml         # bölmeler, örnek sayıları, tohumlar
│   └── train_m2.yaml           # model ve eğitim ayarları
├── src/tpslib/
│   ├── config.py               # YAML → dataclass, birim kontrolü
│   ├── dtau.py                 # analitik D(τ)
│   ├── fit_tps.py              # M0: klasik TPS uydurması
│   ├── sim/
│   │   ├── geometry.py         # r–z ağı, katmanlar
│   │   ├── solver.py           # sonlu hacim, kapalı zaman adımı
│   │   └── sensor_model.py     # halka kaynakları, sabit akım geri beslemesi, uzunluk ağırlıklı okuma
│   ├── measurement/
│   │   ├── acquisition.py      # temiz eğri → cihazın örnekleme hızında ölçüm (gürültü, kayma, ADC)
│   │   └── preprocess.py       # gerçek ve sentetik veri için ORTAK ön işleme
│   ├── data/
│   │   ├── prior.py            # parametre örnekleme + otomatik ayar kuralı
│   │   ├── generate.py         # paralel üretim, HDF5 parçaları, manifest
│   │   ├── features.py         # log-zaman ızgarası, θ ve dθ/d(ln t)
│   │   └── dataset.py          # PyTorch Dataset, eğitim anında rastgele ölçüm etkileri
│   ├── models/
│   │   ├── resnet1d.py         # M2
│   │   └── losses.py           # Gauss negatif log-olabilirlik
│   ├── train/
│   │   └── trainer.py
│   └── eval/
│       ├── metrics.py
│       ├── calibration.py
│       └── report.py
├── scripts/                    # komut satırı girişleri (her biri --config alır)
│   ├── validate_sim.py
│   ├── generate_dataset.py
│   ├── train_m2.py
│   └── evaluate.py
├── tests/
├── data/                       # git'e girmez
│   └── synthetic/v1/           # train.h5, val.h5, test.h5, test_interp.h5, test_faulty.h5, manifest.json
├── runs/                       # git'e girmez; her eğitim kendi klasöründe
├── models/                     # yayınlanan modeller + model kartları
├── notebooks/
├── docs/                       # PROGRESS.md, DECISIONS.md
└── reports/                    # faz raporları ve grafikler
```

## Faz 0 — İskelet ve ortam

- Yukarıdaki yapıyı, `pyproject.toml` dosyasını (numpy, scipy, h5py, pyyaml, torch, matplotlib, pandas, tqdm, lightgbm, onnx, pytest) ve `.venv` ortamını kur.
- Bana PowerShell'de ortamı nasıl etkinleştireceğimi ve testleri nasıl çalıştıracağımı tek satırlık komutlarla yaz.
- CUDA varsa onu, yoksa CPU'yu otomatik seçen bir yardımcı fonksiyon yaz.
- Kapı: `pytest` boş da olsa hatasız çalışır.

## Faz 1 — Analitik çözüm ve M0

- `dtau.py`: Gustafsson D(τ), m parametreli. PROJECT_RULES.md'deki m = 20 test değerlerini %0,05 içinde tutturan birim testleri yaz. Hızlı kullanım için önceden hesaplanmış tablo + interpolasyon ekle ve tablonun doğrudan hesaba göre hatasını test et.
- `fit_tps.py`: ΔT = A·D(τ) + B uydurması, 0,3 < τ² < 1 penceresi, α taraması (en doğrusal sonucu veren α), k = P / (π^1.5·a·A).
- Kapı: analitik formülle üretilmiş gürültüsüz eğrilerde k ve α hatası < %0,5; 0,05–30 W/m·K aralığında 20 farklı k için test.

## Faz 2 — Simülatör ve doğrulama (en kritik faz)

- Eksenel simetrik (r–z) sonlu hacim modeli: iki numune, aralarında sensör katmanları. İnce katmanları (PI, yapıştırıcı, bakır) toplu ısıl direnç + ısı kapasitesi olarak modelle; her iki yüzde ayrı temas iletkenliği h_c; dış yüzeylerde küçük ısı taşınım katsayısı. Ağ sensör düzlemine doğru sıklaşsın; zaman adımı logaritmik büyüsün (1 ms'den başla); seyrek LU ile kapalı çözüm.
- Isı kaynağı: m = 20 halka, Joule ısısı halka uzunluğuyla orantılı. Sabit akım geri beslemesi: P(t) = I²·R0·(1 + α_R·ΔT_ort).
- Çıktı: logaritmik zaman ızgarasında temiz ΔT_ort(t) ve P(t).
- `scripts/validate_sim.py` şu kapıları kontrol edip rapor yazsın:
  1. **Analitik sınır:** katmansız, h_c çok büyük, sabit güç, çok büyük numunede sonuç D(τ) çözümüne %0,5 içinde.
  2. **Yakınsama:** ağ ve zaman adımı yarıya indirildiğinde sonuç %0,1'den az değişir.
  3. **Enerji korunumu:** verilen enerji ile depolanan enerji %0,5 içinde.
  4. **Fiziksel davranış:** h_c düştükçe görünür k düşer; numune incelince geç zamanda sapma başlar; katmanlar erken zamanda sapma yaratır. Her biri için birer grafik.
  5. **Hız:** tek simülasyon < 1 s (tek çekirdek). Ölçülen süreyi ve 60.000 simülasyon için tahmini toplam süreyi raporla.
- Bu fazın sonunda kesinlikle dur; grafikleri birlikte inceleyeceğiz.

## Faz 3 — Prior, otomatik ayar kuralı ve veri üretimi

- `prior_v1.yaml` (bağımsız örneklenir):
  - k: 0,05–30 W/m·K, log-düzgün
  - ρc_p: 0,8–4,5 MJ/m³·K, düzgün (α = k / ρc_p)
  - h_c (alt ve üst ayrı): 500–50.000 W/m²·K, log-düzgün
  - katman kalınlıkları: nominalin ±%20'si; PI ve yapıştırıcı k'si: 0,12–0,25 W/m·K
  - numune yarıçapı 20–30 mm, kalınlığı 8–25 mm
  - α_R ve a için kalibrasyon hatası: ±%1 ve ±%0,5 (normal)
- **Otomatik ayar kuralı** (`prior.py`, cihaz firmware'iyle aynı mantık): k ve α'nın ±%30 hatalı bir "yoklama tahmini" üzerinden; τ = 1'de 2,5 K hedefleyen güç (8 mW – 1 W'a kırpılır); t_son = min(a²/α, ilerleme derinliği numune boyutunun %90'ını aşmayacak süre); veri hızı süreye göre (uzun → 20 SPS, kısa → 1200 SPS). Güç ve süre bu kuraldan gelir, bağımsız rastgele seçilmez.
- Üretim (`generate_dataset.py`): çok süreçli, ilerleme çubuklu, yarıda kalırsa kaldığı yerden devam eden, HDF5 parçalarına yazan. Her örnek: tüm parametreler, temiz ΔT(t), P(t), zaman ızgarası. `manifest.json`: config özeti, tohumlar, git commit, örnek sayıları, üretim süresi.
- Önce 1.000 örneklik pilot: süreyi ölç, parametre dağılımlarının ve ΔT_max dağılımının histogramlarını çiz, bana göster. Onayımdan sonra tam üretim:

| Bölme | Örnek | Tohum | Not |
| --- | --- | --- | --- |
| train | 50.000 | 1001 | |
| val | 5.000 | 2002 | model seçimi |
| test | 5.000 | 3003 | yalnızca en sonda |
| test_interp | 2.000 | 4004 | k ∈ [1,5 ; 2,5] W/m·K bandından; ablasyon deneyinde bu bant train/val'dan çıkarılır |
| test_faulty | 2.000 | 5005 | hatalı ölçümler: çok düşük h_c (hava boşluğu), çok ince numune, kayma sıçraması; güven sistemi testi için |

## Faz 4 — Ölçüm etkileri, ön işleme ve özellikler

- `acquisition.py`: temiz eğriyi cihazın veri hızında örnekle; beyaz + 1/f gürültü (düzeyi `device.yaml`'dan; şimdilik ADS1263 veri sayfasına dayalı yer tutucu, sonra gerçek cihaz kaydıyla değiştirilecek), doğrusal + küçük ikinci derece kayma (±0,2 mK/s), t_c kayması, güç ölçüm hatası (±%0,5), elektronik kayma eğrisi (yer tutucu).
- `preprocess.py`: ham sinyalden ΔT ve P'ye; gerçek cihaz verisi de **aynı fonksiyondan** geçecek.
- `features.py`: 256 noktalı logaritmik zaman ızgarasına binleyerek yeniden örnekleme; θ = ΔT·π^1,5·a/P ve dθ/d(ln t) kanalları; skaler girdiler: ln t_son, ln P, ΔT_max, numune kalınlığı ve yarıçapı, akım aralığı.
- `dataset.py`: ölçüm etkileri **her epokta yeniden rastgele** uygulanır (veri zenginleştirme). Normalizasyon istatistikleri yalnızca train'den, dosyaya kaydedilir.
- Kapı: aynı temiz eğriden üretilmiş 5 farklı gürültülü örneği ve özellik kanallarını gösteren grafik; normalizasyon sonrası kanalların dağılımı; ideal eğri + gürültüde M0'ın hâlâ < %2 hatayla çalıştığını gösteren test.

## Faz 5 — M2 eğitimi

- `resnet1d.py`: 1B ResNet, 4 blok (32–64–128–128 kanal, çekirdek 5, GELU), genel ortalama havuzlama, skalerlerle birleşen 128 nöronlu baş; çıktılar: log k, log α, log h_c için ortalama ve log-varyans. Parametre sayısını yazdır (hedef yaklaşık 200 bin).
- Kayıp: Gauss negatif log-olabilirlik (β-NLL, β = 0,5); hedef ağırlıkları log k 1, log α 0,5, log h_c 0,2.
- Ayarlar (`train_m2.yaml`): AdamW, öğrenme oranı 1e-3, kosinüs azalma + 2 epok ısınma, ağırlık azalması 1e-4, yığın 256, en fazla 60 epok, val log k NLL'sine göre erken durdurma (sabır 8), gradyan kırpma 1,0, karışık hassasiyet yalnızca CUDA'da.
- Önce tek tohumla kısa deneme (5 epok), öğrenme eğrilerini göster. Sonra 5 tohumla topluluk.
- Her koşu `runs/<tarih>_<isim>_seed<n>/` altında: config kopyası, git commit, normalizasyon dosyası, en iyi ağırlıklar, epok bazında metrik CSV'si, öğrenme eğrisi grafiği.
- Ablasyon: aynı ayarlarla, `test_interp` bandı train/val'dan çıkarılmış bir model de eğit (tek tohum yeterli).

## Faz 6 — Değerlendirme ve dışa aktarma

- `evaluate.py`, test setlerinde şunları raporlasın:
  - k için medyan ve 95. yüzdelik bağıl hata, her k aralığında ayrı (0,05–0,5 / 0,5–5 / 5–30 W/m·K); aynısı α için. Hedef: k medyan < %1, 95. yüzdelik < %4.
  - Hata–k ve hata–h_c saçılım grafikleri.
  - Belirsizlik kalibrasyonu: %68 ve %95 aralıklarının gerçek kapsama oranı.
  - M0 ile M2 karşılaştırması aynı test setinde.
  - `test_interp`: ablasyon modelinin görmediği bantta hatası.
  - `test_faulty`: topluluk belirsizliği ve basit kontrollerle hatalı ölçümlerin ne kadarının işaretlenebildiği.
- En iyi topluluğu ONNX'e aktar, PyTorch ile ONNX çıktılarının aynı olduğunu test et. Ayrıca yaklaşık 60 bin parametrelik kompakt bir M2'yi (cihaz için) eğit ve büyük modelle farkını raporla.
- `models/m2_v1/MODEL_CARD.md`: eğitim verisi, geçerli k ve α aralığı, test sonuçları, bilinen sınırlar. Açıkça yaz: "Bu sonuçlar yalnızca simülasyona dayanır; gerçek cihaz verisiyle doğrulama henüz yapılmadı."

Şimdi Faz 0 ile başla.
