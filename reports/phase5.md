# Faz 5 raporu (ara) — M2 eğitim kodu ve pilot denemesi (2026-10-18)

> Bu rapor **kodun hazır olduğunu** gösterir. Sayılar 1.000 örneklik pilot verisinden gelir (900 train / 100 val). Model başarısının ölçüsü değildir. Asıl eğitim, Colab'daki 64 bin örneklik üretim bitince yapılacak.

## Durum

| | |
|---|---|
| Kod | `models/resnet1d.py`, `models/losses.py`, `train/trainer.py`, `scripts/train_m2.py`, Colab hücreleri 7–12 |
| Parametre sayısı | **375.078** (4 blok 32–64–128–128, çekirdek 5, GELU, blok başına 2 evrişim; Faz 0 kararıyla tarif edildiği gibi) |
| Testler | **131 passed** (6 yeni eğitim testi: boyutlar, NLL minimumu, β-NLL gradyanı, epok örnekleyici, uçtan uca eğitim + devam, ablasyon) |
| Pilot denemesi | 40 epok, yığın 32, yerel RTX 3050 Ti: epok ~1 s. k medyan **%1,8**, %95'lik **%14**; α medyan %9; %95 aralığı kapsama oranı **%96** |

## Gerçeklenen ayarlar (`configs/train_m2.yaml`)

- **Model:** global ortalama havuzlama sonrası 7 skalerle birleşen 128 nöronlu baş. Çıktılar ln k, ln α ve ln G_c için ortalama ve log-varyans. Log-varyans yumuşak sınırlı [−14, 4].
- **Kayıp:** β-NLL (β = 0,5); hedef ağırlıkları ln k 1, ln α 0,5, ln G_c 0,2.
- **Optimizasyon:**
  - AdamW, öğrenme oranı 1e-3, ağırlık azalması 1e-4.
  - 2 epok ısınma + kosinüs (adım bazında).
  - Gradyan kırpma 1,0; yığın 256; en fazla 60 epok.
  - Val ln k NLL'sine göre erken durdurma, sabır 8.
- **Karışık hassasiyet:** yalnızca CUDA'da. A100/L4'te bf16, diğerlerinde fp16 + GradScaler.
- **Veri:**
  - Ölçüm etkileri train'de her epokta yeniden çekiliyor. Epok numarası örnek indeksine kodlanıyor (`EpochSampler`), böylece kalıcı DataLoader işçileri de her epokta yeni gürültü üretiyor.
  - Val'de etkiler sabit.
  - Normalizasyon yalnızca train'den (5.000 örnek).
  - Topluluk çeşitliliği için her tohumun gürültü tohumu da farklı.
- **Koşu klasörü** `runs/<etiket>_m2_seed<n>/`:
  - Kayıt: `config.yaml`, `run_info.json` (git commit, kod özeti, cihaz, veri dosyaları), `normalizer.json`.
  - Ağırlıklar: `best.pt`, `last.pt`.
  - İzleme: `metrics.csv`, `learning_curve.png`, `summary.json`.
  - **Aynı komut yarıda kalan koşuyu son epoktan sürdürüyor** (test edildi).
- **Ablasyon:** `--exclude-k-band 1.5 2.5` bandı train'den **ve** val'den çıkarıyor (test edildi).
- **Veri miktarı çalışması:** `--max-train 10000/25000` (Faz 6'daki "daha fazla veri gerekir mi" kararı için).

## Pilot gözlemleri

- 256'lık yığınla 900 örnek epok başına yalnızca 3 adım ediyor; 5 epokluk ilk deneme anlamlı değildi. Yığın 32 ve 40 epokla model düzenli öğrendi (`runs/pilot-trial-b32_m2_seed1/learning_curve.png`). 50 bin örnekte 256'lık yığın epok başına ~195 adım eder; sorun olmaz.
- Val ln k NLL'si 40 epokta hâlâ düşüyordu: bu veri boyutunda model veri sınırlı.
- β-NLL kayıp değeri erken düzleşiyor (β ağırlığı ölçeği bastırıyor). İzleme ve erken durdurma bu yüzden fiziksel birimlerde NLL ve hatayla yapılıyor.

## Sıradaki adımlar (Colab, üretim bitince)

1. Yeni zip'i Drive'a yükleyin. **Üretim bitmeden yüklemeyin**, çünkü değişen kod yarıda kalan parçaları reddettirir.
2. GPU çalışma zamanı (L4 yeterli) → hücre 1, 2, 7 → 8 (5 epok deneme): öğrenme eğrisini birlikte inceleyelim.
3. Onaydan sonra 9 (veri miktarı), 10 (5 tohumlu topluluk), 11 (ablasyon).
