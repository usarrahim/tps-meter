# Faz 2 raporu — Simülatör ve doğrulama (2026-10-17)

> Tüm sonuçlar **simülasyondur**. Bu faz simülatörün sayısal olarak doğru olduğunu ve beklenen fiziği ürettiğini gösterir. Simülatörün gerçek cihazı ne kadar iyi temsil ettiği ancak gerçek ölçümlerle (plan Bölüm 9.7) anlaşılacak.

## Kapı özeti

| Kapı | Hedef | Sonuç | Durum |
|---|---|---|---|
| 1. Analitik sınır (katmansız, mükemmel temas, sabit güç, 80 mm numune) | %0,5 içinde | 0,1 ≤ τ ≤ 1'de en fazla **%0,26**; M0 k hatası ≤ %0,018, α ≤ %0,056 | ✔ |
| 2. Yakınsama (ağ ve zaman adımı yarıya) | < %0,1 değişim | 0,1 ≤ τ ≤ 1'de **%0,093–0,099**; M0 k ≤ %0,019, α ≤ %0,042 | ✔ (sınıra yakın) |
| 3. Enerji korunumu | %0,5 içinde | ayrık denge **~10⁻¹⁶**; P(t)'nin bağımsız yamuk integraliyle de **< 10⁻¹⁵** | ✔ |
| 4a. h_c düşünce görünür k düşer | monoton | 3 malzemede monoton (çelikte h_c=500'de −%10) | ✔ |
| 4b. İnce numunede geç sapma | görünür | 8 mm: τ≈0,59'dan sonra, τ=1'de +%2,6–3,0; 16 mm ve üstünde < %0,01 | ✔ |
| 4c. Katmanlar erken zamanda sapma | görünür | eğim sapması τ ≈ 0,13–0,78'e kadar %1'i aşıyor (h_c'ye bağlı); sonra sabit kayma kalıyor | ✔ |
| 5. Hız | < 1 s | **2,1–2,6 s** (tek çekirdek) | ✘ |

**Hız kapısı geçmedi.** Doğruluk öncelikli olduğu için bilerek böyle bırakıldı (gerekçe aşağıda). Tek çekirdekte 60.000 simülasyon yaklaşık 39 saat sürer. Paralel çalışmada bu dizüstünde **bellek bant genişliği** sınır koyuyor (Faz 3 pilotu): verim 6–12 süreçte ~1,5–1,6 simülasyon/s'de doyuyor, yani 64.000 örnek **~11–12 saat**.

Testler: **78 passed** (13'ü simülatör testi). Yeniden üretmek için: `python scripts/validate_sim.py`. Çıktılar `reports/phase2/` altında.

## Simülatörün yapısı

- **Geometri:** Eksenel simetrik r–z düzlemi, iki numune.
- **Sensör katmanları:** Numuneler arasında dört katman var: PI taban 25 µm | bakır seviyesi 18 µm (izlerde bakır, aralarda yapıştırıcı) | yapıştırıcı 15 µm | PI örtü 12,5 µm.
  - Her katman toplu direnç ve ısı kapasitesi olarak modelleniyor, kendi içinde radyal iletim var.
  - Alt ve üst yüzde ayrı h_c tanımlı.
  - Film Ø20 mm; film dışında numuneler arasında hava boşluğu var.
  - Numunelerin dış yüzeylerinde h = 10 W/m²·K taşınım var.
- **Kaynak ve okuma:** 20 halka, Joule ısısı halka uzunluğuyla orantılı. Okunan sıcaklık uzunlukla ağırlıklı ortalama. Sabit akım geri beslemesi P = I²R₀(1 + α_R·ΔT) adım içinde kapalı (implicit) çözülüyor.
- **Zaman:** TR-BDF2 (γ = 2−√2; L-kararlı, 2. mertebe). İlk seviye, güç basamağını sönümlemek için Richardson ekstrapolasyonlu kapalı Euler ile yapılıyor.
  - TR-BDF2'nin iki aşaması aynı matrisi kullanıyor: adım başına 2 geri çözüm (Richardson'da 3).
  - dt her 12 adımda ikiye katlanıyor; seviye başına tek LU ayrıştırması yeterli.
  - Seyrek LU (SuperLU, simetrik minimum derece sıralaması) kullanılıyor.
- **Çıktı:** 512 noktalı log-zaman ızgarasında ΔT(t) ve P(t), ayrıca enerji dökümü.

## Ağ çalışmasında öğrenilenler (önemli)

1. **Bakır iz kenarında ~20 µm'lik bir sınır tabakası var.** Bakır, komşu yapıştırıcı ve PI'ye yanal ısı sızdırıyor; tabakanın genişliği √(k_katman·t/U) ≈ 22 µm.
   - İz başına 1 hücreli kaba ağ bu tabakayı çözemiyor. Sonuç sabit bir ΔT kayması: PMMA'da %1,5, çelikte **%15**.
   - Çözüm: kenarda 1,5 µm hücre ve 1,15 büyüme.
   - Bu kayma k'yı neredeyse etkilemiyor, ama yüksek k'da ince katman etkisinin büyüklüğünü belirliyor. h_c tahmini ve gerçek veriye uyum için önemli.
2. **Uzak alanın büyüme oranları geç zaman şeklini, dolayısıyla α'yı belirliyor.** Sensör dışında radyal büyüme ve derinlikte z büyümesi için 1,15 yerine 1,05 seçildi. Bu, M0 α hatasını %0,9'dan %0,02'ye indirdi.
3. **İki bölgeli ağ.** İnce sütunlar yalnızca sensör düzleminin 0,3 mm yakınında, uzakta kaba sütunlar kullanılıyor.
   - İki bölge doğrusal interpolasyonlu, simetrik ve korunumlu bir arayüzle bağlı.
   - İlk denediğim parçalı sabit arayüz k'da %1–5 hata verdi ve reddedildi.
   - Bu yapı aynı doğrulukta maliyeti yaklaşık 4 kat düşürüyor.
4. **α zaman adımına duyarlı.** Richardson'lı Euler'de seviye başına 8 adımda ΔT'deki fark yalnızca %0,01, ama M0 α %0,1 kayıyor; 16 adım gerekiyordu. Faz 3'te paralel verimi artırmak için TR-BDF2'ye geçildi. 12 adım, 64 adımlık referansa göre α'yı %0,02 içinde tutuyor ve çözüm sayısı yarıya iniyor.
5. **Sonuç:** < 1 s hedefiyle %0,1 yakınsama aynı anda sağlanamadı. Doğruluk tercih edildi. Daha kaba bir ağ (1 s civarı) çelikte %15 kayma ve %0,2 α hatası veriyordu.

## Fiziksel bulgular (M2 neden gerekli)

`physics_hc.png` ve `physics_thickness.png` ile `physics_layers.png` grafiklerine göre, klasik M0 gerçekçi koşullarda sistematik hata yapıyor:

| Durum | M0 k hatası | M0 α hatası |
|---|---|---|
| PMMA, h_c = 5000 | −%0,07 | −%0,9 |
| Borosilikat, h_c = 5000 | −%0,8 | +%0,3 |
| Paslanmaz 304, h_c = 5000 | **−%1,7** | **+%3,0** |
| Paslanmaz 304, h_c = 500 (kötü temas) | **−%10** | **+%28** |
| PMMA, 12 mm numune | +%0,9 | **−%12** |
| PMMA, 8 mm numune | −%6 | −%55 |

Yorum:
- **Yüksek k'lı malzemelerde katmanlar ve temas M0'ı bozuyor.** Bu, Zhang ve diğerlerinin düşük k için bildirdiği etkinin tersi yönde: yüksek k'da sensörün kendi ısıl direnci baskın hâle geliyor. M2'nin bu etkiyi öğrenmesi gerekiyor; h_c ve katman parametreleri prior'da bu yüzden var.
- **α, k'dan çok daha hassas.** Numune biraz inceldiğinde ΔT yalnızca %0,2 sapıyor, ama α %12 kayıyor. Güven sistemindeki ilerleme derinliği kontrolü (plan 9.8) bu yüzden kritik.
- **Sınır kalınlık.** 16 mm ve üstünde sonlu boyut etkisi < %0,01. Prior'daki 8–25 mm aralığı hatalı ölçüm örneklerini de kapsıyor.

## Grafikler ve tablolar

- `phase2/gate1_analytic.png` — simülatör, sonlu iz genişlikli analitik çözüm ve D(τ) + sabit; bağıl fark.
- `phase2/gate2_convergence.png` — ağ + zaman ve yalnız zaman yarıya inince değişim.
- `phase2/physics_hc.png`, `physics_thickness.png`, `physics_layers.png` — fiziksel davranış.
- Tablolar: `gate1_analytic.csv`, `gate2_convergence.csv`, `gate3_energy.csv`, `physics_*.csv`, `speed.csv`, `summary.json`.

## Sınırlar ve açık noktalar

- **Türev pürüzsüzlüğü.** dΔT/d ln t'nin türevinde, zaman seviyesi geçişlerinde ve kaydın son noktalarında, eğimin ≤ %1'i kadar küçük sıçramalar var (`physics_layers.png`'deki ince kırılmalar). M2'nin dθ/d ln t kanalında ölçüm gürültüsü bunun çok üstünde olacak.

- **Erken zaman (τ < 0,1).**
  - Analitik sınırda fark en erken anlarda %1–4'e çıkıyor (sonlu bant çözümünün çok ince yakın alanı).
  - Yarıya inme değişimi τ ≈ 0,03'te %0,2'ye kadar çıkıyor.
  - Kapılar 0,1 ≤ τ ≤ 1 penceresinde tanımlandı. M2 eğrinin tamamını görecek; bu erken bölgedeki %0,2'lik sayısal belirsizlik ölçüm gürültüsünün altında kalıyor, ama yazılı olarak not ediyorum.
- **Modellenmeyen fizik.** Uç (lead) ısınması (plan: R_uç/R₀ ≈ %0,4), PETG plaka ve sıcaklığa bağlı malzeme özellikleri modelde yok. Bunlar plan 9.7'deki artık karşılaştırmasında gerekirse eklenecek.
- **Hız.**
  - Veri üretimi için kabul edilebilir: 60 bin simülasyon bir gece sürer.
  - M0+ (simülatörle ters çözüm, her ölçümde yüzlerce simülasyon) için yavaş. Orada daha kaba bir "hızlı ağ" ve son adımda bu ağ kullanılabilir; Faz 6 sonrası ele alınacak.
