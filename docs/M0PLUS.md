# M0+ — yeniden simülasyonlu ters çözüm ve artık kontrolü (tasarım, henüz uygulanmadı)

Plan 9.8'in son güven kontrolü. Amaç: M2'nin tahminiyle simülatörü yeniden çalıştırıp ölçülen eğriyi gürültü düzeyinde açıklayıp açıklayamadığını görmek. Açıklayamıyorsa ölçüm veya model alanı dışında bir şey var (hava boşluğu, aşırı asimetrik temas, modellenmeyen fizik).

Güven v2'deki kontroller (aralık, topluluk farkı, M0 farkı) özellikle **aşırı asimetrik temasta** zayıf kalıyor: M2 bu eğrilere de kendinden emin, aralık içinde bir cevap veriyor. M0+ bu boşluğu kapatmak için gerekli.

## Bilinmeyenler

| Parametre | Tür | Başlangıç | Nasıl çözülür |
|---|---|---|---|
| ln k, ln α | doğrusal olmayan | M2 tahmini | Levenberg–Marquardt (LM) |
| ln h_c,alt, ln h_c,üst | doğrusal olmayan | M2'nin G_c'sinden simetrik bölüşüm | LM (iki ayrı parametre) |
| B (sabit kayma + kalan doğrusal kayma eğimi) | doğrusal | — | her LM adımında en küçük karelerle kapalı çözüm (variable projection) |
| t_c (ısıtma başlangıç kayması) | doğrusal olmayan ama ucuz | 0 | simüle eğrinin zaman ekseninde kaydırılması (spline), yeni simülasyon gerekmez |

- İki h_c'yi ayrı bırakmak önemli: uyan h_c,maks/h_c,min oranı doğrudan "asimetri > 10" kontrolü olur.
- Film katmanı kalınlıkları ve özellikleri prior ortalamasında sabit tutulur (gerçek cihazda bir kez kalibre edilir).

## Algoritma (ölçüm başına)

1. Ön işlenmiş kayıt (`preprocess` çıktısı: t, ΔT, P) M2 ile aynı yoldan gelir.
2. Kalıntıyı 256 log-zaman bininde hesapla (özelliklerle aynı ızgara). Her binin gürültü σ_i'si ön ölçüm taban gürültüsünden ve bin başına örnek sayısından gelir.
3. LM: 4 doğrusal olmayan parametre, ileri farkla Jakobiyen (adım 1e-3, ln uzayında). t_c ve B her değerlendirmede iç içe çözülür.
4. Durma: göreli χ² iyileşmesi < 1e-3 veya 8 iterasyon.
5. Çıktılar:
   - χ²_ν = ortalama((r_i/σ_i)²) → **artık kontrolü**, eşik val'ın %99'luk diliminden.
   - Uyan h_c oranı → **asimetri kontrolü** (> 10 ise arıza).
   - Uyan k, α → M0+ tahmini. M2 ile farkı da bir kontrol.

## Süre tahmini

- Tek simülasyon (Faz 2 hız ölçümü, bu dizüstü, tek süreç): 2,1–2,6 s (`reports/phase2/speed.csv`).
- LM iterasyonu: 1 merkez + 4 Jakobiyen simülasyonu = 5 simülasyon. M2 başlangıcı iyi olduğu için 3–5 iterasyon, artı 1–2 reddedilen adım → **~20–30 simülasyon**.
- **Ölçüm başına: tek çekirdekte ~50–75 s.** Jakobiyen 5 çekirdeğe paralel dağıtılırsa ~12–18 s duvar süresi.
- Hızlandırma seçenekleri:
  - Kaba ağ (yarı radyal/eksenel çözünürlük) ilk iterasyonlarda, ince ağ son iterasyonda: ~2 kat.
  - "M0+ hafif": yalnızca M2 tahminiyle tek bir simülasyon (~2,5 s) ve χ²_ν. Hava boşluğunu yakalaması beklenir; asimetriyi yakalamaz (h_c bölüşümü bilinmiyor).
- Cihaz kullanımı için uygun: ölçümün kendisi 5–800 s sürüyor; PC tarafında 15–75 s ek süre kabul edilebilir.

## Doğrulama planı (Colab)

- Eşik: val'dan 500 normal ölçüm (~500 × 60 s ≈ 8 CPU-saat → 24 çekirdekte ~20 dk).
- Etki: test_faulty'den her arıza türünden 200, test'ten 300 asimetri > 10 örneği.
- Test setlerinin üçüncü kullanımı olacağı için önce val içinde asimetri > 10 örnekleriyle değerlendirilmeli; test yalnızca son raporda.
- Gerçek veride ilk kullanım: PMMA ve camda χ²_ν ≈ 1 çıkıyorsa simülatör gerçek eğriyi yapı bırakmadan açıklıyor demektir (plan 9.7 sim-to-real kontrolüyle aynı test).
