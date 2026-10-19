# Faz 6 raporu — Değerlendirme ve dışa aktarma (2026-10-19)

> **Bu sonuçlar yalnızca simülasyona dayanır; gerçek cihaz verisiyle doğrulama henüz yapılmadı.** Simülasyon testindeki başarı gerçek başarı değildir. Proje başarı kriteri olan "≤ %15, gerçek referans malzemelerde" ayrı ve sonraki aşamada ölçülecek.

Test setleri (`test`, `test_interp`, `test_faulty`) burada **bir kez** kullanıldı. Model seçimi ve tüm eşikler doğrulama setiyle yapıldı. Model kartı: [`models/m2_v1/MODEL_CARD.md`](../models/m2_v1/MODEL_CARD.md). Ham çıktılar: `reports/phase6/` (`summary.json`, CSV'ler, grafikler).

## Kapı özeti

| Madde (prompt) | Sonuç | Durum |
|---|---|---|
| k medyan < %1, %95'lik < %4, her k bandında | 0,05–0,5: %0,83 / %2,60 · 0,5–5: %0,89 / %3,17 · 5–30: %2,07 / %10,36 | ✔ ✔ ✘ |
| α aynı bantlarda | %0,71 / %2,47 · %0,92 / %6,81 · %2,78 / %9,32 | raporlandı |
| Hata–k ve hata–h_c grafikleri | `test_error_scatter.png` | ✔ |
| %68 / %95 aralıkların kapsaması | ln k: %74,7 / %97,0 · ln α: %89,3 / %99,3 · ln G_c: %69,1 / %96,4 | ✔ (k ve α temkinli) |
| M0–M2 karşılaştırması | Tümü k: M2 %1,01 / %5,85 · M0 %1,71 / %13,47 | ✔ M2 belirgin üstün |
| test_interp (ablasyon) | Bandı görmeyen model %0,92 / %2,91; gören %0,86 / %2,58 | ✔ |
| test_faulty işaretleme (plan hedefi %90) | İnce numune %97,9 · kayma sıçraması %84,9 · hava boşluğu %52,8 | ✘ kısmen |
| ONNX, PyTorch ile aynı | En büyük fark 2·10⁻⁶ | ✔ |
| Kompakt model (~60 bin) | 2 × 61.910 parametre: k %1,08 / %6,14; toplulukla fark medyan %0,27 | ✔ |
| Model kartı | `models/m2_v1/MODEL_CARD.md` | ✔ |

## Yorum

1. **Ana hedef aralıkta (k < 5 W/m·K) simülasyon hedefi karşılandı.** Projenin ana referans seti (PMMA 0,19, PTFE 0,25, borosilikat 1,2) bu aralıkta.
   - Hata, kalibrasyon hatalarından gelen kaldırılamaz tabana (~%0,77 medyan) çok yakın.
   - Modelin kendisi bu bantta darboğaz değil. Gerçek doğruluğu α_R kalibrasyonu, güç ölçümü, temas ve referans değerlerin kendi belirsizliği belirleyecek.
2. **Yüksek k (5–30 W/m·K) hedefi karşılanmadı.** %95'lik hata %10, medyan %2.
   - Faz 2'den beri bilinen fiziksel zorluk: güç 1 W'ta kırpılıyor, sinyalin çoğu sensörün kendi katmanlarından geliyor.
   - Yine de klasik yönteme göre 2–2,5 kat daha iyi. M0'ın bu banttaki sistematik −%5 sapması M2'de yok (`test_error_scatter.png`, orta panel).
   - Plandaki doğrulanmış aralık zaten 0,19–16 W/m·K. Paslanmaz çelik (16) bu bandın içinde.
3. **M0'ın sistematik hatası belirgin.** Düşük k'da +%2–3, yüksek k'da −%5 ile −%30. Bu sapma sensör katmanlarından ve temastan geliyor; gerçek cihazda da beklenmeli. M2'yi ve planlanan M0+'yı (yeniden simülasyonlu ters çözüm) değerli kılan bu.
4. **Ara değerlere genelleme iyi.** Görmediği bantta ablasyon modeli neredeyse ana model kadar doğru (%0,92 vs %0,86). Belirsizliği de kalibre kaldı (%95 kapsama %96). Yeni bir malzemenin "eğitimde görülmemiş k"si sorun değil.
5. **Belirsizlik kalibrasyonu:**
   - ln G_c neredeyse ideal.
   - ln k hafif temkinli.
   - ln α belirgin temkinli: %68 aralık gerçek değeri %89 oranında içeriyor. Yani α belirsizliği gereğinden geniş. Val üzerinden tek bir ölçek katsayısıyla düzeltilebilir; güvenli taraf olduğu için acil değil.
6. **Güven kontrolleri tek başına yeterli değil.**
   - İnce numune ve kayma sıçraması iyi yakalanıyor.
   - Hava boşluğu ölçümlerinin yalnızca %53'ü ve aşırı asimetrik temasların %4'ü işaretleniyor.
   - İşaretlenmeyen arızalı ölçümlerde k medyan %14 düşük.
   - Plan 9.8'deki iki kontrol henüz yok: "tahmin edilen G_c eğitim aralığında mı" ve "yeniden simülasyon artığı" (M0+). Gerçek ölçüme geçmeden önce güven sisteminin tamamlanması gerekiyor.

## Faz 5'ten aktarılan bulgular

- **Temas asimetrisi kararı.** Kullanıcı kararıyla iki yüz arasında 10 kattan büyük temas farkı arıza sayıldı. Eğitim alanı buna göre süzüldü (37.604 örnek). Bu, gerçekçi temas alanında yüksek k'nın %95'lik hatasını ~%13'ten ~%9'a, α medyanını %3,7'den %1,5'e indirdi (Faz 5 teşhisi). Gerçek temas dağılımı ilk ölçümlerden öğrenilince veri v2 ile yeniden değerlendirilecek.
- **Topluluk tutarlılığı.** 5 üye val'da birbirine çok yakın: k medyan %1,05–1,12, %95'lik %6,1–6,4. Eğitim kararlı.

## Sonraki adımlar (öneri)

1. **Güven sistemini tamamlamak** (gerçek ölçümden önce):
   - G_c ve k aralık kontrolleri
   - M0+: simülatörle doğrusal olmayan en küçük kareler ters çözümü + yeniden simülasyon artığı
   - α belirsizlik ölçeğinin val'dan kalibrasyonu
2. **Cihaz tarafı** (plan İP1–İP4): sensör siparişi, ölçüm zinciri devreye alma (T1–T4).
3. **İlk gerçek veriler** (sim-to-real, plan 9.7):
   - Boşta ve sahte yük kayıtları → `device.yaml` gerçek değerleri.
   - PMMA ve camda ilk ölçümler → simülatörün gerçek eğriyi artık yapı bırakmadan üretip üretmediğinin kontrolü ve gerçek temas dağılımı.
   - Ardından gerekirse veri v2 ve yeniden eğitim; sonra ince ayar.
