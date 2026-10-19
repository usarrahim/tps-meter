# M2 v1 — model kartı

> **Bu sonuçlar yalnızca simülasyona dayanır; gerçek cihaz verisiyle doğrulama henüz yapılmadı.**
> Gerçek doğruluk ancak referans malzemelerle, malzeme düzeyinde çapraz doğrulamayla (plan T8) ve kilitli test malzemeleriyle (T9) belirlenecek.

| | |
|---|---|
| Sürüm | M2 v1 (base model, yalnızca simülasyonla eğitildi) |
| Tarih | 2026-10-19 |
| Topluluk | 5 üye (`v1sym_m2_seed1…5`), üye başına 375.078 parametre |
| Kompakt cihaz modeli | 2 üye (`v1c_m2c_seed1…2`), üye başına 61.910 parametre |
| Kod | git `6d03bf1` ve sonrası; veri v1 manifest'i `data/synthetic/v1/manifest.json` |

## Ne yapar

Bir TPS (hot-disk) ölçüm eğrisinden üç büyüklüğü, her birinin belirsizliğiyle birlikte tahmin eder:
- **k:** ısıl iletkenlik
- **α:** ısıl yayınım
- **G_c:** sensörden iki numuneye toplam arayüz iletkenliği

Belirsizlik iki parçanın toplamıdır: model belirsizliği ve 5 üye arasındaki fark.

**Kullanım zinciri** (gerçek ve sentetik veri için aynı):
1. Ham cihaz kaydı (ADC oranı, akım, R₀, kayma) → `tpslib.measurement.preprocess`
2. → `tpslib.data.features.make_features`: 3 kanal × 256 log-zaman bini + 7 skaler
3. → `m2_ensemble.onnx`: normalizasyon grafın içinde. Çıktılar ln birimlerinde: `mean`, `std` ve üye tahminleri `member_mean`.

```python
import numpy as np, onnxruntime as ort
from tpslib.measurement.preprocess import preprocess
from tpslib.data.features import make_features
f = make_features(preprocess(raw_record))
sess = ort.InferenceSession("models/m2_v1/m2_ensemble.onnx")
mean, std, members = sess.run(None, {"seq": f["seq"][None], "scalars": f["scalars"][None]})
k, k_lo, k_hi = np.exp(mean[0, 0]), np.exp(mean[0, 0] - 1.96 * std[0, 0]), np.exp(mean[0, 0] + 1.96 * std[0, 0])
```

ONNX ile PyTorch arasındaki en büyük fark 2·10⁻⁶ (ln birimi).

## Eğitim verisi

- **Kaynak:** sentetik veri v1, 64.000 simülasyon, 0 hata.
- **Simülatör:** r–z sonlu hacim. 20 halkalı bakır spiral (a = 6,4 mm). PI / bakır+yapıştırıcı / yapıştırıcı / PI film katmanları. İki yüzde ayrı temas iletkenliği. Sabit akım geri beslemesi. Doğrulama kapıları: `reports/phase2.md`.
- **Ölçüm parametreleri:** güç, süre ve veri hızı cihazın otomatik ayar kuralından gelir.
- **Eğitim kümesi:** temas asimetrisi (h_c,max/h_c,min) ≤ 10 olan 37.604 örnek.
- **Ölçüm etkileri:** her epokta yeniden eklendi. Gürültü, kayma, t_c, güç hatası, elektronik kayma, α_R ve yarıçap kalibrasyon hataları (`configs/device.yaml`, şu an yer tutucu değerler).

## Geçerli aralık

| Parametre | Aralık |
|---|---|
| k | 0,05–30 W/m·K |
| ρc_p | 0,8–4,5 MJ/m³·K (α ≈ 0,011–37 mm²/s) |
| Numune | kalınlık 8–25 mm, yarıçap 20–30 mm, iki özdeş parça |
| Temas | her yüzde h_c 500–50.000 W/m²·K, iki yüz arasında en fazla 10 kat fark |
| Ölçüm | güç 8 mW–1 W, veri hızı 20–1200 SPS, τ_son ≈ 0,5–1,2 |

## Test sonuçları (kilitli test seti, gerçekçi temas alanı, n = 3.715)

| k bandı [W/m·K] | n | M2 k medyan | M2 k %95 | M0 k medyan | M0 k %95 | M2 α medyan | M2 α %95 |
|---|---|---|---|---|---|---|---|
| 0,05–0,5 | 1.392 | **%0,83** | **%2,60** | %1,21 | %3,65 | %0,71 | %2,47 |
| 0,5–5 | 1.346 | **%0,89** | **%3,17** | %1,41 | %6,39 | %0,92 | %6,81 |
| 5–30 | 977 | %2,07 | %10,36 | %5,03 | %23,86 | %2,78 | %9,32 |
| Tümü | 3.715 | %1,01 | %5,85 | %1,71 | %13,47 | %1,03 | %7,32 |

- **Simülasyon hedefi** (k medyan < %1, %95'lik < %4) **k < 5 W/m·K'de karşılandı.** 5–30 bandında karşılanmadı.
- **Kaldırılamaz taban:** kalibrasyon hataları (α_R σ %1, yarıçap σ %0,5) ve ±%0,5 güç hatası eğriden görülemez. Yalnız bunlar k'da ~%0,77 medyan ve ~%2,3 %95'lik hata yaratır.
- **Belirsizlik kalibrasyonu:** %95 aralığı gerçek değeri ln k'da %97,0, ln α'da %99,3, ln G_c'de %96,4 oranında içeriyor. ln α belirsizliği gereğinden geniş (temkinli).
- **İnterpolasyon:** k = 1,5–2,5 bandını hiç görmeyen ablasyon modeli o bantta %0,92 / %2,91 hata verdi. Bandı gören model %0,86 / %2,58. Ara değerlere genelleme iyi.
- **Kompakt model** (2 × 61.910 parametre): tümünde k %1,08 / %6,14. Büyük toplulukla fark medyan %0,27, %99'luk %2,1.

## Güven kontrolleri (v1)

Eşikler doğrulama setinin %99'luk dilimlerinden belirlendi. Kullanılan kontroller: topluluk farkı, toplam belirsizlik, M0–M2 farkı, M0 artığı ve ΔT aralığı.

| Durum | İşaretlenen |
|---|---|
| İnce numune (2–6 mm, yanlış girilen kalınlık) | %97,9 |
| Kayma sıçraması (5–50 mK) | %84,9 |
| Hava boşluğu (h_c 20–300) | **%52,8** |
| Temas asimetrisi > 10 (eğitim alanı dışı) | **%4,2** |
| Normal ölçüm (yanlış alarm) | %4,7 |

## Bilinen sınırlar

1. **Yalnızca simülasyon.** Gerçek cihazın gürültüsü, kayması, t_c'si, elektronik kayması ve gerçek temas dağılımı yer tutucu değerlerle modellendi. Gerçek veri geldiğinde `device.yaml` güncellenip model yeniden eğitilecek, gerekirse veri v2 üretilecek (plan 9.7).
2. **Yüksek k (> 5 W/m·K) en zor bölge.** Güç 1 W'ta kırpılıyor ve ölçülen sıcaklık artışının çoğu sensörün kendi katmanlarından ve temastan geliyor. Bu bantta %95'lik hata ~%10.
3. **Kötü ve asimetrik temas yeterince yakalanmıyor.** Hava boşluğu ölçümlerinin yarısı, aşırı asimetrik temasların çoğu işaretlenmiyor. İşaretlenmeyen arızalı ölçümlerde k ~%14 düşük çıkıyor. Plan 9.8'deki iki kontrol henüz yok:
   - Tahmin edilen G_c'nin eğitim aralığında olup olmadığı
   - Tahmini parametrelerle yeniden simülasyon artığı (M0+)

   Bunlar eklenene kadar düşük k sonucu temas kontrolü yapılmadan kullanılmamalı.
4. **α belirsizliği temkinli.** Gerçekleşen kapsama beklenenden yüksek. Val üzerinden bir ölçek katsayısıyla daraltılabilir.
5. **Modellenmeyen fizik:** uç (lead) ısınması, PETG fikstür plakası, anizotropi, sıcaklığa bağlı malzeme özellikleri, yüzey pürüzlülüğü, sıvılar ve gözenekli malzemeler.
6. **Tek sensör geometrisi:** varyant A (a = 6,4 mm, m = 20, b = 0,10 mm). B/C varyantları veya farklı film yapısı için yeniden veri üretimi ve eğitim gerekir.

## Dosyalar

| Dosya | Açıklama | Git |
|---|---|---|
| `m2_ensemble.onnx` | 5 üyeli topluluk + normalizasyon (7,6 MB) | hayır (Drive: `tps-meter/phase6_bundle.zip`) |
| `m2c_compact.onnx` | 2 üyeli kompakt cihaz modeli (0,5 MB) | hayır |
| `normalizer_v1sym_m2_seed*.json` | Üye normalizasyonları (yalnızca train'den) | hayır (model dosyası, gizli) |
| PyTorch ağırlıkları | Drive: `tps-meter/runs/v1sym_m2_seed*/best.pt`, `v1c_m2c_seed*/best.pt` | hayır |
