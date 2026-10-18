# Faz 4 raporu — Ölçüm etkileri, ön işleme ve özellikler (2026-10-18)

> Ölçüm zinciri modeli **yer tutucu** değerlerle çalışıyor: ADS1263/1262 veri sayfası ve proje planı. Gerçek cihazın boşta kaydı (plan T2/T3) gelince `device.yaml` güncellenecek. Bunun için **veri yeniden üretilmez**, çünkü etkiler eğitim sırasında ekleniyor. Bu fazdaki sonuçlar simülasyona dayanır.

## Kapı özeti

| Kapı | Sonuç | Durum |
|---|---|---|
| Aynı temiz eğriden 5 gürültülü örnek ve özellik kanalları | `phase4/realisations_and_channels.png` (düşük ve yüksek k) | ✔ |
| Normalizasyon sonrası dağılımlar | 3 kanal ve 7 skalerin hepsi ortalama 0, std 1, sonlu (`normalised_distributions.png`) | ✔ |
| M0, ideal eğri + ölçüm etkileri, < %2 | 20 k × 5 tekrar: en büyük k hatası **%0,70**; kalibrasyon hataları da eklenince **%1,59** | ✔ |
| Testler | **125 passed** (Faz 4'e ait 34 yeni test) | ✔ |

Normalizasyon burada pilot verisiyle denendi. Gerçek istatistikler Faz 5'te **yalnızca train** üzerinden hesaplanıp modelle birlikte kaydedilecek. Kod, train dışı bir bölmeden istatistik çıkarılmasını reddediyor; bunun da testi var.

## Ne yapıldı

### `measurement/acquisition.py`: temiz eğri → cihazın ham kaydı

Sentetik çıktı, gerçek cihazın kaydettiği alanlarla aynı:
- ADC1 oranı R/R_ref (32 bit niceleme)
- ADC2 akımı (10 SPS)
- Ön ölçümden R₀ ve kayma eğimi
- Kalibre edilmiş α_R ve yarıçap

Eklenen etkiler (`device.yaml`, her çağrıda rastgele):

| Etki | Model |
|---|---|
| Gürültü | Beyaz: 0,23 µV × √(SPS/20) × (1–2 kat, plan T2 sınırı). Sıcaklık karşılığı V/(I·R₀·α_R); 100 mA ve 20 SPS'te 0,15 mK, planla aynı. 1/f köşesi 0–1 Hz. |
| Ortam kayması | ±0,2 mK/s doğrusal + küçük kuadratik. Ön ölçümün eğim tahmininde 5 µK/s hata. Ön işleme yalnızca tahmin edilen doğrusal kısmı çıkarabiliyor. |
| Zaman | Isıtma başlangıcı kayması t_c ±5 ms, saat hatası 20 ppm |
| Güç | ±%0,5 mutlak akım/güç hatası, ADC2 gürültüsü |
| Kalibrasyon | α_R (σ %1) ve yarıçap (σ %0,5) prior'dan; R₀ ölçüm hatası; R_ref toleransı (oranlı ölçümde sadeleşiyor, test edildi) |
| Elektronik kayma | R_ref kendi ısınması 10 mK/W, τ 20 s; düzeltme eğrisi ±%50 hatalı |
| Arıza | test_faulty için kayma sıçraması |

### `measurement/preprocess.py`: ortak ön işleme (plan 9.2)

- Adımlar: R = oran × R_ref → ΔT = (R/R₀ − 1)/α_R → ön ölçümdeki kayma çıkarılır → elektronik kayma eğrisi çıkarılır → P = I²·R.
- **Gerçek cihaz verisi de aynı fonksiyondan geçecek.**
- Etkiler kapatılınca zincir temiz eğriyi 10⁻⁷ K içinde geri veriyor (test edildi).

### `data/features.py`

- **Izgara:** [t_son/1000, t_son] aralığında 256 log-zaman bini. Seyrek binlerde değerler binin ortalama ln t konumundan bin merkezine interpolasyonla taşınıyor.
- **Kanallar** (üçüncüsü benim eklemem):
  - `θ/θ_ref`
  - `dθ/d ln t / θ_ref` (Savitzky–Golay, 15 bin)
  - `ln(1 + bin başına örnek)`
  - Gerekçeler aşağıda.
- **Skalerler:** ln θ_ref, ln t_son, ln P₀, ΔT_max, girilen kalınlık ve yarıçap, akım aralığı.
- **Hedefler:** ln k, ln α, ln G_c.

### `data/dataset.py`

- PyTorch Dataset, HDF5'ten temiz eğri okuyor.
- **Train'de etkiler her epokta yeniden çekiliyor** (tohum = taban, epok, örnek). Val ve test'te örnek başına sabit, böylece metrikler tekrarlanabilir.
- HDF5 her işçi sürecinde ayrı açılıyor (Windows uyumlu).
- Ablasyon için k bandı dışlama (`select_indices`).
- `Normalizer`: yalnızca train'den hesaplanır, JSON olarak modelle saklanır.
- **Hız:** örnek başına 4,2 ms. 50 bin örnekli bir epok tek çekirdekte 3,5 dk, 8 işçiyle yaklaşık 0,5 dk.

## Tasarım kararları ve gerekçeleri

1. **θ'yı θ_ref'e bölüp ln θ_ref'i skaler yapmak.** θ ≈ D(τ)/k + sabit, yani 0,05–30 W/m·K aralığında 600 kat değişiyor. Ham hâliyle verilirse ağ, büyüklüğü öğrenmeye uğraşırken şekli kaçırır. Bu ayrımla **bilgi kaybı yok**: genlik ln θ_ref'te, şekil kanalda. ΔT/ΔT_max'tan farkı da bu; orada genlik ve dolayısıyla k bilgisi kayboluyordu.
2. **Örnek sayısı kanalı (prompta ek).** 20 SPS'teki uzun ölçümlerde erken binler boş ya da tek örnekli, geç binlerde yüzlerce örnek var. Kanal, modele her binin ne kadar gürültülü olduğunu söylüyor. Bilgi tamamen ölçüm düzeninden geliyor, sızıntı yok.
3. **Ön ölçüm kayma eğimini hatalı tahmin ediyor ve kuadratik kayma çıkarılamıyor.** Gerçekte de böyle; model bu artığa dayanıklı olmayı öğrenmeli.

## M0'ın ölçüm etkilerine duyarlılığı

Her etki tek başına, 20 tekrarın en kötüsü (`phase4/m0_effect_sensitivity.csv`):

| Etki | k=0,05 (820 s, 20 SPS) | k=0,19 (350 s, 20 SPS) | k=1,2 (85 s, 100 SPS) | k=16 (9 s, 1200 SPS) |
|---|---|---|---|---|
| **Ortam kayması** | k %0,51 / **α %10,3** | k %0,24 / **α %4,9** | k %0,06 / α %1,2 | k %0,02 / α %0,3 |
| Gürültü | k %0,31 / α %1,8 | k %0,21 / α %1,8 | k %0,07 / α %0,4 | k %0,12 / α %0,8 |
| Güç hatası | k %0,46 / α %0,0 | k %0,46 / α %0,0 | k %0,46 / α %0,0 | k %0,46 / α %0,0 |
| Kalibrasyon (α_R %1, a %0,5) | k %0,58 / α %1,0 | aynı | aynı | aynı |
| Elektronik kayma | ~0 | ~0 | k %0,06 / α %0,2 | k %0,23 / α %1,7 |
| t_c, saat | ~0 | ~0 | ~0 | k %0,16 / α %0,3 |

Yorum:
- **k sağlam:** her etki %0,6'nın altında; plandaki belirsizlik bütçesiyle (10.2) uyumlu.
- **α'yı uzun, düşük k'lı ölçümlerde ortam kayması belirliyor (%5–10).** Planın "PMMA'da ortam kayması %1–2" uyarısıyla tutarlı; α eğri şekline k'dan çok daha duyarlı. M2 kaymayı eğitimde gördüğü için bu konuda M0'dan iyi olmalı; Faz 6'da ölçeceğiz.
- **Kalibrasyon hatalarını hiçbir yöntem düzeltemez.** α_R'deki %1 doğrudan k'ya geçiyor (yarıçapla kısmen sadeleşiyor). α_R kalibrasyonu (plan T4) bu yüzden kritik.

## Açık noktalar (gerçek cihazla netleşecek)

- **t_c (ısıtma başlangıcı kayması) ±5 ms yer tutucu.** Yüksek k'da (1200 SPS) ilk ~30 ms'yi belirgin biçimde değiştiriyor (`realisations_and_channels.png`, alt satır). DAC komutu ile DRDY zaman damgası aynı zamanlayıcıdan alınırsa gerçek değer çok daha küçük olur. Firmware bitince ölçülmeli.
- Gürültü düzeyi, 1/f köşesi, kayma istatistiği ve elektronik kayma eğrisi cihazın boşta ve sahte yük kayıtlarından alınacak (plan 9.7).
- **Önemli:** `configs/device.yaml` değişti. Colab'daki üretim bitmeden **yeni zip yüklemeyin.** Değişen config, yarıda kalan parçaları "farklı kod" diye reddettirir.
