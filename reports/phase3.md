# Faz 3 raporu — Prior, otomatik ayar kuralı ve veri üretimi: pilot (2026-10-17)

> Bu veri yalnızca **simülasyondur** ve temiz eğrilerden oluşur. Gürültü, kayma ve kalibrasyon hataları Faz 4'te eğitim sırasında eklenecek.

## Özet

| | Sonuç |
|---|---|
| Pilot | 1.000 örnek (tohum 999), **1.000 başarılı / 0 hata** |
| Enerji dengesi (en kötü örnek) | 7·10⁻¹¹ |
| Süre | 693 s duvar saati, 10 süreç → **1,44 örnek/s** |
| **Tam üretim tahmini** (64.000 örnek: train 50k + val 5k + test 5k + test_interp 2k + test_faulty 2k) | **~12,3 saat** (bu dizüstünde, 10 süreç) |
| ΔT_max | medyan 2,56 K, %5–95 aralığı 1,58–5,24 K; < 0,5 K: %0; > 8 K: %0,6 |
| Testler | 90 passed |

Grafikler `reports/phase3/` altında:
- `pilot_params.png`: parametre dağılımları.
- `pilot_autosetup.png`: güç, süre, veri hızı, τ_son, akım aralığı.
- `pilot_dTmax.png`: ΔT_max dağılımı.
- `pilot_curves.png`: örnek eğriler.
- `pilot_timing.png`: simülasyon süreleri.

Sayılar `pilot_summary.json` dosyasında.

## Yapılanlar

- **`data/prior.py`**
  - prior_v1'e göre bağımsız örnekleme. Her film katmanının kalınlık çarpanı ayrı çekiliyor.
  - Tüm çekilişler sabit sırayla ve vektörel yapılıyor. Aynı tohum, işçi sayısından bağımsız olarak aynı parametreleri veriyor.
  - **Otomatik ayar kuralı** (cihaz firmware'iyle aynı mantık, plan 10.3):
    - Yoklama tahmini: k ve α her biri U(−%30, +%30) hatalı.
    - Güç: τ = 1'de 2,5 K hedefi (ideal TPS formülüyle), 8 mW–1 W'a kırpılıyor.
    - Süre: `min(a²/α_yoklama, 2√(α_yoklama·t) = 0,9·min(kalınlık, yarıçap − a))`.
    - Veri hızı: kayıt başına 12.000 örnek bütçesinde izin verilen en yüksek hız (plan 7.4).
    - Akım aralığı: plan 5.3'teki L/H aralıkları.
    - Hesaplarda kalibre edilmiş sensör yarıçapı kullanılıyor.
  - **Model hedefi `G_c`:** bakırdan iki numuneye toplam arayüz iletkenliği (katman dirençleri + h_c, iki yol paralel). Alt ve üst h_c ayrıca saklanıyor.
  - **test_faulty için arızalar:** her örnekte tam olarak bir arıza var.
    - Hava boşluğu: bir veya iki yüzde h_c = 20–300 W/m²·K.
    - İnce numune: gerçek kalınlık 2–6 mm, ama kullanıcı prior'daki kalınlığı girmiş; otomatik ayar bu yanlış değeri kullanıyor.
    - Kayma sıçraması: 5–50 mK; yalnızca etiket ve parametre olarak saklanıyor, sıçrama Faz 4'teki ölçüm zincirinde eklenecek.
- **`data/generate.py` + `scripts/generate_dataset.py`**
  - Çok süreçli üretim (Windows için `spawn`), ilerleme çubuğu var.
  - 250 örneklik HDF5 parçaları atomik olarak yazılıyor (geçici dosya + yeniden adlandırma).
  - **Yarıda kalırsa aynı komut kaldığı yerden devam ediyor.**
  - Bölme bitince parçalar `<bölme>.h5` dosyasında birleştiriliyor.
  - Her örnekte saklananlar: tüm parametreler, 512 noktalı log-zaman ızgarasında (1/SPS'ten t_son'a) temiz ΔT(t) ve P(t), simülasyon tanı bilgileri. Hatalı simülasyon üretimi durdurmuyor, işaretleniyor.
  - `manifest.json` içeriği: tüm config'ler + sha256, tohumlar, git commit (henüz commit yok, `null`), kod özeti (src + configs sha256), sayılar ve süreler.
- **Faz 2 simülatöründe değişiklik:** zaman şeması TR-BDF2'ye geçti (gerekçe aşağıda). Faz 2 kapıları yeni şemayla yeniden çalıştırıldı, hepsi geçti; `reports/phase2.md` güncellendi.
- **Hata düzeltmesi:** PyYAML `0.8e6` gibi işaretsiz üslü sayıları metin olarak okuyordu. Yükleyici düzeltildi; config'lerde sayısal metin kalmadığını doğrulayan test eklendi. Önceki sonuçlar etkilenmedi, çünkü kod bu değerleri zaten `float()` ile çeviriyordu.

## Paralel üretim süresi: neler denendi

| Değişiklik | Verim (örnek/s) | 64k tahmini |
|---|---|---|
| Faz 2 şeması (Richardson-Euler, 16 adım), 14 süreç | 1,15 | ~15,5 sa |
| Aynı şema, 8 süreç (1–16 arası tarandı) | 0,96 | — |
| **TR-BDF2, 12 adım, 6–12 süreç** | **1,44–1,59** | **~11–12 sa** |

- Tek başına 2,3 s süren bir simülasyon, 10 süreç paralel çalışırken 6,9 s sürüyor. Darboğaz CPU değil, **bellek bant genişliği**: seyrek LU geri çözümleri büyük üçgen matrisleri bellekten okuyor.
- TR-BDF2 adım başına çözüm sayısını yarıya indirdi; doğruluk aynı (Faz 2 kapıları yeniden geçti).
- 8 süreçten sonra verim artmıyor.

## Pilot bulguları

1. **Dağılımlar beklendiği gibi.** k log-düzgün, ρc_p düzgün; α 0,013–20 mm²/s. G_c 1.000–16.000 W/m²·K, ortası ~6.000.
2. **Otomatik ayar.**
   - Örneklerin %21'inde güç 1 W'ta kırpılıyor (k ≳ 7 W/m·K), %2'sinde 8 mW'ta.
   - Ölçümlerin %43'ünde süreyi τ = 1 değil, numune boyutu (ilerleme derinliği) sınırlıyor.
   - Gerçek τ_son 0,52–1,21; yoklama hatası yüzünden bazı ölçümler τ = 1'i aşıyor.
   - Veri hızları: 20 SPS %31, 50 SPS %12, 100 SPS %23, 400 SPS %16, 1200 SPS %19.
3. **ΔT_max'ı film/temas ofseti büyütüyor.** Kural ideal formülle 2,5 K hedefliyor, ama sensörün kendi ısıl direnci ve h_c bir ofset ekliyor.
   - Yüksek k ve düşük G_c'de ΔT_max 6–10 K'e çıkıyor; %0,6'sı güven sistemindeki 8 K sınırını aşıyor.
   - **Açık soru:** gerçek firmware gücü yoklama darbesinde *ölçülen* ΔT'ye göre seçerse bu ofset otomatik olarak telafi edilir. Promptta istenen "yoklama k'sı + ideal formül" kuralını uyguladım. Firmware tasarlanırken hangi kural kullanılacaksa simülatörde de o olmalı.
4. **Yüksek k zor bölge olacak.** k = 30 örneğinde normalize eğrinin çoğu film ofsetinden geliyor; k'yı taşıyan geç zaman eğimi küçük (`pilot_curves.png` sağ panel). Faz 6'da 5–30 W/m·K bandında hatanın büyük çıkmasını bekliyorum. Plan da doğrulanmış aralığı 0,19–16 W/m·K olarak veriyor.

## Tam üretim için seçenekler

1. **Hepsi bu bilgisayarda:** ~12 saat, 10 süreç, bir gece. Komut:
   `python scripts/generate_dataset.py --splits train val test test_interp test_faulty --workers 10`
   Yarıda kalırsa aynı komut devam ettirir.
2. **Bölüşmeli:** train (50k, ~9,5 sa) bu bilgisayarda; val, test, test_interp ve test_faulty (14k) aynı anda Colab'da. Toplam süre ~9,5 saate iner. Bölmeler ayrı tohumlu ve deterministik olduğu için sonuç aynı olur; iki manifest sonra birleştirilir. Colab CPU çalışma zamanının hızı henüz ölçülmedi.

**Onayınız olmadan tam üretimi başlatmıyorum.**
