# Faz 1 raporu — Analitik çözüm ve M0 (2026-10-17)

> Bu fazdaki tüm sonuçlar **analitik, gürültüsüz sentetik veriye** dayanır. Bu sonuçlar kodun doğru olduğunu gösterir (planın T5 testi); cihazın gerçek doğruluğunu göstermez.

## Özet

| Kapı | Hedef | Sonuç | Durum |
|---|---|---|---|
| D(τ), PROJECT_RULES.md'deki 6 değer | %0,05 içinde | en büyük sapma **%0,0007** | ✔ |
| Tablo + interpolasyon, doğrudan hesaba göre | — | en büyük mutlak fark **2,6·10⁻¹³** | ✔ |
| M0, 20 k (0,05–30 W/m·K), sabit güç | k ve α < %0,5 | k ≤ %0,0002, α ≤ %0,001 | ✔ |
| M0, aynı 20 k, sabit akım (gerçek cihaz koşulu) | k ve α < %0,5 | k ≤ **%0,003**, α ≤ **%0,019** | ✔ |
| pytest | hatasız | **65 passed** | ✔ |

Tek uydurmanın süresi en fazla 0,07 s (74.000 örneklik PMMA kaydı dahil).
Yeniden üretmek için: `python scripts/validate_m0.py`. Çıktılar: `reports/phase1/`.

## Yapılanlar

- `src/tpslib/dtau.py`
  - Gustafsson D(τ), m parametreli; taşmayı önlemek için `i0e` kullanılıyor.
  - Doğrudan (uyarlamalı kuadratür) hesap ve hızlı tablo: ln σ ızgarasında Simpson + kübik spline. Tablo 0,16 s'de bir kez kuruluyor.
  - İdeal sıcaklık artışı iki durum için: sabit güç, ve **sabit akım** (P = I²R₀(1+α_R·ΔT), Duhamel süperpozisyonuyla çözülüyor).
- `src/tpslib/fit_tps.py` (M0)
  - Model: ΔT = A·D̃(t) + B·P(t)/P₀.
  - D̃, güç geçmişine göre ağırlıklı D'dir. Sabit güçte D̃ = D(τ) olur ve model klasik ΔT = A·D(τ) + B'ye indirgenir.
  - Pencere 0,3 < τ² < 1. α taraması: önce kaba log tarama, sonra pencere sabitken 1B en küçükleme ve pencere güncelleme, pencere değişmeyene kadar.
  - k = P₀ / (π^1,5·a·A).
  - Sonuçla birlikte artık RMS, pencere, τ_max ve "pencere kesildi" uyarısı dönüyor.
- `src/tpslib/synth_analytic.py`: test eğrileri.
  - Gerçek k ve α ile basitleştirilmiş otomatik ayar: τ=1'de 2,5 K hedefleyen güç (8 mW–1 W'a kırpılır), τ=1'e kadar süre.
  - Veri hızı: kayıt başına 12.000 örnek bütçesinde izin verilen en yüksek hız.
- `configs/analysis.yaml`: D(τ) ve M0 ayarları. `device.yaml`'a örnek bütçesi eklendi.
- Testler: `tests/test_dtau.py`, `tests/test_fit_tps.py`.

## Önemli bulgu 1 — D(τ)'nun alt sınırı

- D(τ) integralinin integrandı σ→0'da sabit bir değere (1/(2√π(m+1)) = 0,0134) yaklaşıyor (ln σ cinsinden). Bu yüzden integral **logaritmik olarak ıraksıyor**: sonsuz ince bir halkanın kendi üzerindeki sıcaklığı sonsuzdur (`dtau.png` sağ panel).
- PROJECT_RULES.md'deki referans değerler tam olarak **σ_min = 10⁻⁶** alt sınırıyla elde ediliyor. Bu değer config'e yazıldı.
- Sonuç olarak D'nin mutlak değeri σ_min'e bağlı. Bağımlılık yalnızca sabit bir terim olarak giriyor. M0'da bu terim B'ye gidiyor; k ve α bundan hiç etkilenmiyor (`test_cutoff_independent`).
- Faz 2'ye etkisi: simülatör analitik çözümle mutlak ΔT üzerinden değil, şu iki yoldan biriyle karşılaştırılacak:
  - sonlu iz genişliği b ile düzenlenmiş bir referans,
  - veya k ve α'nın geri kazanımı.
- Ayrıntı: `docs/DECISIONS.md`.

## Önemli bulgu 2 — Sabit akımda güç değişimi

Cihaz sabit akımla ölçüyor; güç ısınmayla yaklaşık %1 artıyor. Bu artış yok sayılıp ortalama güç kullanılırsa (`m0_constant_current_effect.png`):
- k **%0,66–0,73**,
- α **%1,9–2,2**

sistematik olarak düşük çıkıyor. Bu hata tek başına simülasyon hedefinin (k medyan < %1) büyük kısmını harcar. M0 bu yüzden ölçülen P(t)'yi kullanıyor; hata %0,003'e iniyor.

## Grafikler

- `phase1/dtau.png`: D(τ) ve referans noktalar; σ→0 davranışı.
- `phase1/m0_gate_errors.png`: 20 k değerinde M0'ın k ve α hatası, iki güç modunda.
- `phase1/m0_constant_current_effect.png`: P(t) hesaba katılmazsa oluşan hata.
- `phase1/m0_example_fit.png`: PMMA benzeri örnek; ΔT–D̃ doğrusu ve pencere içi artık (±20 µK, eğri interpolasyonundan).
- Tablolar: `phase1/dtau_reference.csv`, `phase1/m0_gate.csv`, `phase1/summary.json`.

## Sınırlar ve sonraki adımlar

- M0 henüz t_c (başlangıç zamanı kayması) uydurmuyor. Faz 4'te gürültü, kayma ve t_c eklenince M0'ın < %2 kapısı orada sınanacak. Gerekirse t_c parametresi eklenecek.
- Gerçek ölçümde B, Kapton katmanlarındaki sıcaklık farkını taşır. Erken zamanda katman ısı kapasitesinin yarattığı sapma Faz 2 simülatörüyle görülecek; pencerenin (0,3 < τ²) bu sapmayı dışarıda bırakıp bırakmadığı orada ölçülecek.
