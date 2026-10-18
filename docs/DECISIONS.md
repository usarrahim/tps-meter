# Tasarım kararları

Her karar: tarih, karar, gerekçe. Açık sorular kullanıcı yanıtlayınca karara dönüşür.

## 2026-10-17 — Faz 0

- **Python 3.11 (yerel), ≥3.11 uyumlu kod.** Bilgisayarda yalnızca 3.14 vardı; PROJECT_RULES.md gereği Python install manager ile 3.11.9 kuruldu. Colab şu an 3.12 kullandığı için `requires-python = ">=3.11"`; 3.12'ye özgü sözdizimi kullanılmayacak.
- **Yerel PyTorch CUDA 12.8 tekerleğiyle** (RTX 3050 Ti 4 GB, sürücü CUDA 13.1). `pyproject.toml` torch'u sürümsüz listeler; yerelde önce CUDA tekerleği kurulur, Colab kendi torch'unu kullanır.
- **Proje kökü `tps-master/`** (prompttaki `tps-meter/` adı yerine mevcut klasör).
- **Veri ve çıktı kökü ortam değişkeniyle değişebilir** (`TPS_DATA_ROOT`, `TPS_RUNS_ROOT`, `src/tpslib/runtime.py`). Colab'da veri yerel diske, çıktılar Drive'a.
- **Cihaz seçimi** `tpslib.runtime.get_device()`: CUDA varsa CUDA, yoksa CPU; `TPS_DEVICE=cpu|cuda` ile zorlanabilir.
- **Yer tutucu değerler:** `device.yaml` gürültü modeli ve akım aralıkları, `sensor.yaml` yapıştırıcı/PI nominal ρc_p'si. Gerçek değerler gelince değiştirilecek; dosyalarda `PLACEHOLDER` olarak işaretli.

## 2026-10-17 — Faz 0 açık sorularının kararları

Kullanıcı yanıtı: "Plan kurgusunu kendin yap, küçültme derdimiz yok, en başarılı şekilde, doğruluk öncelikli, Drive'a zip ile."
Proje planı PDF'i (Tasarım ve Uygulama Planı, Bölüm 2, 9, 10) referans alındı.

1. **h_c çıktısı → toplam arayüz iletkenliği `G_c`.** Sensör yalnızca ortalama sıcaklığı okuduğu için alt ve üst h_c ayrı ayrı zayıf tanımlanır; tanımlanabilir olan, bakırdan iki numuneye toplam ısıl iletkenliktir:
   `G_c = 1/(R_alt_katman + 1/h_alt) + 1/(R_üst_katman + 1/h_üst)` [W/m²·K]. Model `log G_c` çıkarır (ağırlık 0,2). Alt/üst h_c veri setinde ayrı saklanır, gerekirse sonradan hedef yapılabilir.
2. **M2 mimarisi tarif edildiği gibi** (4 blok, 32–64–128–128, çekirdek 5, blok başına 2 konvolüsyon, ≈330 bin parametre). "~200 bin" yalnızca tahmindi; küçültme yapılmıyor. Kompakt cihaz modeli (~60 bin) planda kalıyor ama masaüstü sonucunu etkilemiyor.
3. **test_faulty:** dosyada temiz eğri + arıza etiketi; kayma sıçraması değerlendirmede sabit tohumla aynı `acquisition` yolundan eklenir.
4. **Enerji dengesi:** `∫P dt = Σ (ρc_p·V·ΔT)_tüm hücreler (numuneler + katmanlar + bakır) + ∫ dış yüzey taşınım kaybı dt`, bağıl fark < %0,5. Ayrıca numunelerde depolanan pay ayrıca raporlanır.
5. **M0 penceresi 0,3 < τ² < 1** (PDF Bölüm 2: literatür kuralı, Zhang ve diğ.). Config'te ayarlanabilir; ince numunede (cam 10 mm) τ ≤ 0,75'e indirilebilir.
6. **Doğruluk > hız.** Yakınsama kapısı önce gelir; < 1 s hedefi tutmazsa paralel üretimle telafi edilir ve süre raporlanır.
7. **Colab:** kod zip paketiyle (`scripts/pack_for_colab.py`). Önce yerel süre ölçülür; yerel makul ise yerelde çalışılır.

## 2026-10-17 — D(τ) alt integral sınırı (Faz 1)

- **Bulgu:** D(τ) integralinin integrandı σ→0'da `1/(2√π(m+1)σ)` gibi davranır; integral logaritmik ıraksar (sonsuz ince halkanın kendi üzerindeki sıcaklığı sonsuzdur). PROJECT_RULES.md test değerleri (D(0,1)=0,20553 …) alt sınır **σ_min = 1e-6** ile tam olarak elde ediliyor (0,2055295).
- **Karar:** `dtau` alt sınırı config'te `sigma_min: 1.0e-6`. Bu seçim D'ye yalnızca sabit bir terim ekler: `c = ln(1/σ_min)/(2√π(m+1))`. Sabit güçte bu terim B'ye gider; sabit akımda P(t)·c olarak değişir, bu yüzden M0 modeli `ΔT = A·D̃(t) + B·P(t)/P₀` olarak kuruldu (D̃: Duhamel ile güç geçmişine göre ağırlıklı D). Böylece kesme sabiti k ve α'yı hiç etkilemez.
- **Sonuç:** Mutlak "ideal" ΔT (`P/(π^1.5·a·k)·D`) σ_min'e bağlıdır, fiziksel değildir. Faz 2'de simülatör–analitik karşılaştırması, sonlu iz genişliği b ile düzenlenmiş referansa veya (A, B) uydurmasına göre yapılacak, mutlak D değerine göre değil.

## 2026-10-17 — Kapsam genişletme (araştırmacıya yönelik çıktı)

Kullanıcı "araştırmacıların işine tam yarayacak çıktı" istedi. PDF Bölüm 9.5 ve 9.8'e göre promptta olmayan şu parçalar eklenecek (Faz 6'dan sonra, ayrı faz olarak önerilecek):
M0+ (simülatörle tam fizik ters çözümü), güven sistemi (ilerleme derinliği, ΔT aralığı, yeniden simülasyon artığı, topluluk ve yöntem uyuşması), sonuç biçimi `k = … ± … W/m·K (%95), α = … ± … mm²/s, durum + neden` (JSON/TXT/PDF).
- PDF'deki "ortam sıcaklığı" skaler girdisi sentetik veride anlamlı simüle edilemediği için M2'ye şimdilik konmuyor; gerçek veriyle ince ayarda eklenecek.

## 2026-10-17 — Faz 2: simülatör tasarımı

- **İnce film yapısı (alttan üste):** PI taban 25 µm | bakır seviyesi 18 µm (izlerde bakır, aralarda kaplama yapıştırıcısı) | yapıştırıcı 15 µm | PI örtü 12,5 µm. Her katman, radyal sütun başına bir düğüm (toplu direnç + kapasite) ve kendi içinde radyal iletim. Film yarıçapı 10 mm (plan: Ø20 mm aktif bölge); film dışında iki numune arasında film kalınlığında hava boşluğu var.
- **Isı kaynağı ve okuma:** bakır kesiti her halkada aynı olduğundan Joule ısısı iz hacmine düzgün dağılıyor (halka gücü ∝ halka uzunluğu). Okuma da aynı ağırlıkla yapılıyor, yani uzunlukla ağırlıklı ortalama.
- **Sabit akım:** P = P₀(1 + α_R·T̄) zaman adımı içinde kapalı (implicit) çözülüyor; tek mertebeli (rank-1) terim Sherman–Morrison ile ele alınıyor.
- **Zaman:** Önce Richardson ekstrapolasyonlu kapalı Euler kullanıldı (16 adım/seviye; 8 adım M0 α'yı ~%0,1 kaydırıyordu). Faz 3 pilotunda paralel üretimin bellek bant genişliğine takıldığı görüldü. Bu yüzden **TR-BDF2'ye (γ = 2−√2)** geçildi: iki aşama aynı matrisi kullanıyor, adım başına 2 çözüm yetiyor, 12 adım/seviye ile aynı doğruluk sağlanıyor. İlk seviye, güç basamağının sert modlarını sönümlemek için kapalı Euler + Richardson ile yapılıyor. dt her seviyede ikiye katlanıyor, seviye başına tek LU yeterli. İlk adım 0,1 ms; yavaş kayıtlarda `dt0 = max(0,1 ms, 0,05·t_ilk_çıktı)`.
- **Ağ (ağ çalışmasının sonucu, `reports/phase2.md`):**
  - Bakır iz kenarlarında yanal sızıntının ~20 µm'lik bir sınır tabakası var. İz kenarında 1,5 µm hücre kullanılıyor; boşlukta 1,15, iz içinde 1,4 büyüme.
  - İki bölge var. Sensör düzleminin 0,3 mm yakınında ince (iz hizalı) sütunlar, uzakta kaba sütunlar. İki bölge doğrusal interpolasyonlu, simetrik ve korunumlu bir arayüzle (PᵀGP) bağlı.
  - Sabit-değer (parçalı sabit) arayüz %1–5 k hatası verdiği için reddedildi.
  - Uzak alanda z büyümesi 1,05; sensör dışında radyal büyüme 1,05. Geç zaman şekli ve α bu iki orana duyarlı.
- **SuperLU sıralaması `MMD_AT_PLUS_A` + `SymmetricMode`:** SPD matris için COLAMD'ye göre yaklaşık %40 daha az doluluk ve 2 kat daha hızlı çözüm.
- **Analitik referans:** D(τ) sonsuz ince halkalar için tanımlı ve mutlak değeri σ_min'e bağlı (Faz 1). Analitik sınır kapısı bu yüzden sonlu iz genişlikli bant çözümüne göre yapılıyor; bu çözüm Hankel dönüşümüyle tam ifadeden (`sim/analytic.py`) geliyor. b → 0'da D(τ) + sabite gittiği test edildi.

## 2026-10-17 — Faz 3: prior ve otomatik ayar

- **Otomatik ayar kuralı:**
  - Güç: τ = 1'de 2,5 K hedefi, yoklama k'sı ve ideal formülle; 8 mW–1 W'a kırpılır.
  - Süre: `min(a²/α_y, ilerleme derinliği = 0,9·min(kalınlık, yarıçap − a))`. İlerleme derinliği kuralı plan 9.8'deki güven kontrolüyle aynı.
  - Veri hızı: kayıt başına 12.000 örnek bütçesinde izin verilen en yüksek hız.
  - Akım aralığı: plan 5.3'teki L/H.
  - Kalibre edilmiş yarıçap kullanılıyor.
  - Not: ideal formül film/temas ofsetini görmediği için yüksek k'da ΔT_max 3–10 K çıkıyor. Firmware gücü ölçülen ΔT'ye göre seçecekse kural güncellenmeli (açık soru, `reports/phase3.md`).
- **Arızalı ölçümler:** her örnekte tek arıza.
  - Hava boşluğu: h_c 20–300.
  - İnce numune: gerçek kalınlık 2–6 mm, girilen değer yanlış.
  - Kayma sıçraması: ölçüm zincirinde eklenecek.
- **Üretim düzeni:** 250 örneklik atomik HDF5 parçaları, devam ettirme, bölme dosyasında birleştirme, manifest. Git commit olmadığı için kod özeti (sha256) de kaydediliyor.
- **Zaman şeması:** paralel üretimde bellek bant genişliği sınırı nedeniyle Richardson-Euler (adım başına 3 çözüm) yerine TR-BDF2 (2 çözüm) seçildi.
- **YAML:** PyYAML'ın `1e6` biçimini metin okuması özel yükleyiciyle düzeltildi.

## 2026-10-18 — Faz 4: ölçüm zinciri ve özellikler

- **Ham kayıt biçimi:** Ölçüm etkileri temiz eğriye doğrudan değil, cihazın ham kayıt biçiminde ekleniyor (ADC1 oranı, ADC2 akımı, ön ölçüm R₀ ve eğim, kalibrasyon). Böylece gerçek veri ile sentetik veri `preprocess` fonksiyonundan birebir aynı yoldan geçiyor.
- **Gürültü modeli:** ADC giriş gürültüsü (µV) üzerinden kuruldu; sıcaklık karşılığı akımla (1/I) ölçekleniyor (plan 5.2 tablosuyla aynı). Gerçek kartın veri sayfasının 2 katına kadar gürültülü olabileceği hesaba katıldı (U(1, 2)).
- **θ ölçekleme:** θ/θ_ref kanal + ln θ_ref skaler. θ, k ile 600 kat değişiyor; bu ölçekleme bilgi kaybetmeden koşullanmayı iyileştiriyor. ΔT/ΔT_max reddedildi (k bilgisini siler); Fourier sayısı girdi yapılamaz (α bilinmiyor).
- **Ek kanal:** ln(1 + bin başına örnek sayısı) eklendi (prompta ek). Modele bin başına gürültü düzeyini söylüyor.
- **Etki tohumları:** Train'de etkiler her epokta yeni (tohum = taban, epok, indeks); val ve test'te sabit.
- **Normalizasyon:** yalnızca train'den; train dışı bölmede hesaplama kodda engellendi.
- **Açık nokta:** t_c ±5 ms yer tutucu; yüksek k'da erken zamanı belirgin değiştiriyor. Firmware'de DAC adımı ile DRDY zaman damgasının ilişkisi ölçülmeli.

## 2026-10-18 — Faz 5: eğitim ve veri alanı

- **Erken durdurma ancak 30. epoktan sonra (`min_epochs: 30`).** Prompttaki "sabır 8" kuralı, 50k koşusunu 28. epokta, öğrenme oranı hâlâ yüksekken ve val metriği gürültülüyken durdurdu. Kosinüs azalmasından sonraki asıl iyileşme dönemi kaçtı. Sabır 8 korunuyor, yalnızca eğitimin ikinci yarısında devreye giriyor.
- **Temas asimetrisi (açık karar, kullanıcıya soruldu):** Prior alt ve üst h_c'yi bağımsız çekiyor; veride 30–100 kat asimetri sık. Yüksek k'da bu örnekler ayırt edilemez bir belirsizlik yaratıyor (k %20–30 düşük tahmin). Asimetri ≤ 10 ile eğitilen model, gerçekçi temas aralığında yüksek k %95'lik hatasını %13'ten %8,6'ya, genel α medyan hatasını %3,7'den %1,5'e indirdi. Karşılaştırma, tüm veriyle eğitilen model tamamlanınca adil hâle gelecek.
- **Düzeltme:** Sabır sayacı artık yalnızca `min_epochs`'tan sonra sayıyor. İlk sürüm sayacı baştan saydığı için devam ettirilen 50k koşusu 30. epokta yine durdu.
- **Karar (kullanıcı, 2026-10-18): temas asimetrisi > 10 arıza sayılır.** Topluluk (`v1sym`) ve ablasyon `--max-asym 10` ile eğitiliyor. Mevcut veri süzülüyor (train 37,6k); gerçekçi temas dağılımı gerçek ölçümlerden öğrenilince v2 veri üretilecek. Tüm veriyle tam karşılaştırma atlandı: (b) modeli kendi alanında yeterince iyi ve karar fiziksel gerekçeye dayanıyor.

## 2026-10-18 — Faz 6: değerlendirme tasarımı

- **Test alanı:** Ana sonuçlar test setinin gerçekçi temas alanında (asimetri ≤ 10) raporlanıyor. Asimetrisi > 10 olan örnekler ayrı bir satırda ("arıza alanı") gösteriliyor ve güven kontrolleriyle işaretlenmeleri bekleniyor.
- **Güven kontrolleri:** Topluluk farkı, toplam belirsizlik, M0–M2 farkı, M0 artığı ve ΔT aralığı. Eşikler **val**'ın %99'luk dilimlerinden; test setleri eşik seçiminde hiç kullanılmıyor. Yeniden simülasyon kontrolü (M0+) bu sürümde yok.
- **ONNX:** Topluluk normalizasyonlarıyla birlikte tek bir graf; girdi ham özellikler, çıktı ln birimlerinde ortalama ve std. PyTorch ile karşılaştırması otomatik yapılıyor.
- **Kompakt model:** (16, 32, 48, 48) kanal, çekirdek 5, baş 64 → 61.910 parametre; plan 9.11'deki gibi 2 üye.
