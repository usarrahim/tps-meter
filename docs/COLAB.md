# Google Colab ile çalışma

Tüm uzun işler (veri üretimi, eğitim) Colab'da çalışır. Kod bilgisayarda geliştirilir.

## Kod nasıl taşınır: Drive (zip)

- **Seçilen yol: zip paketi, Drive üzerinden.** Veri ve eğitim çıktıları zaten Drive'da duracak. Ek hesap, token ya da depo ayarı gerekmiyor.
- **GitHub'a geçmek ne zaman mantıklı?** Ekip arkadaşıyla birlikte geliştirmeye başladığınızda (plan 9.10). Depo özel (private) olacaksa Colab'da kişisel erişim jetonu (token) gerekir. Kod bu durumda `git clone` ile gelir; geri kalan akış aynı kalır.

## Kurulum (bir kez)

1. Drive'da `Drive'ım/tps-meter/` klasörünü oluşturun.
2. Bilgisayarda kod paketini üretin:
   `python scripts/pack_for_colab.py` → `dist/tps-meter-code.zip`
3. Bu zip'i Drive'daki `tps-meter/` klasörüne sürükleyip bırakın. Kod her değiştiğinde paketi yenileyip yeniden yükleyin; eski dosyanın üzerine yazın.
4. Colab'da `notebooks/colab_run.ipynb` dosyasını açın. Defter de zip içinde; ayrıca Drive'a koyup çift tıklayıp "Google Colaboratory ile aç" diyebilirsiniz.

## Çalışma zamanı seçimi (veri üretimi)

- Üretim yalnızca CPU kullanıyor ve **bellek bant genişliğine** bağlı. Bu dizüstünde 6 süreçten sonra hızlanma durdu.
- En çok CPU çekirdeği veren çalışma zamanını seçin. Hangi türün kaç çekirdek verdiği hesaba göre değişebilir; 2. hücre çekirdek sayısını yazdırır.
- 4. hücre (`bench_generation.py`) birkaç işçi sayısını ölçer, en hızlısını ve tam üretimin tahmini süresini yazar. 5. hücredeki `WORKERS` değerini buna göre ayarlayın.
- GPU bu adımda kullanılmıyor. GPU'lu bir çalışma zamanı ancak daha çok CPU çekirdeği veriyorsa işe yarar.

## Oturum kapanırsa

- Parçalar (250 örnek) **Colab'ın yerel diskinde** üretiliyor; bitince anında `Drive'ım/tps-meter/data/synthetic/v1/` altına kopyalanıyor (`--mirror`).
- Oturum koparsa en fazla o an üretilmekte olan parçalar kaybolur (birkaç dakikalık iş).
- **Devam etmek için:** 1, 2 ve 5. hücreleri yeniden çalıştırın. Bitmiş parçalar Drive'dan geri yüklenir, kalanlar üretilir.
- Kod değiştiyse eski parçalar **reddedilir** (code_hash kontrolü). Bu, farklı simülatör sürümlerinin aynı veri setine karışmasını önler.
- Tarayıcı sekmesini açık tutun. Bilgisayar uyku moduna girerse Colab bağlantıyı koparabilir. Bitiş durumunu 6. hücre gösterir.

## Bölmeler

| Bölme | Örnek | Tohum |
|---|---|---|
| train | 50.000 | 1001 |
| val | 5.000 | 2002 |
| test | 5.000 | 3003 |
| test_interp | 2.000 | 4004 |
| test_faulty | 2.000 | 5005 |

Aynı config, kod ve tohum her makinede aynı parametreleri ve aynı temiz eğrileri verir. Paralel işçi sayısı sonucu değiştirmez.

## Eğitim (Faz 5)

Veri Drive'dan Colab yerel diskine kopyalanıp oradan okunacak (Drive'dan doğrudan HDF5 okumak yavaş). Eğitim çıktıları doğrudan Drive'a yazılacak (`TPS_RUNS_ROOT`).

## Güven kontrolleri v2 (Faz 6 sonrası)

1. Bilgisayarda `python scripts/pack_for_colab.py` → `dist/tps-meter-code.zip`; Drive'daki `tps-meter/` klasöründeki eskisinin üzerine bırakın.
2. Colab'da yeni bir defter açın (Faz 6'daki çalışma zamanı yeterli), aşağıdaki hücreyi yapıştırıp çalıştırın. Defterin son hücresi de aynısıdır.
3. Sonunda `TRUST_REPORT.md` ekrana yazılır; bu metni gönderin. Dosyalar: `tps-meter/reports/phase6_trust/`.

Bu koşu test setlerinin **ikinci kullanımıdır** (ilki Faz 6). Eşikler val'dan seçilir.

```python
# 15) Güven kontrolleri v2 + varyans ölçeği (TEST SETLERİNİN 2. KULLANIMI). Tek başına çalışır; ~10 dk.
from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, shutil, zipfile
DRIVE_DIR = pathlib.Path('/content/drive/MyDrive/tps-meter')
WORK = pathlib.Path('/content/tps-meter')
shutil.rmtree(WORK, ignore_errors=True)
zipfile.ZipFile(DRIVE_DIR / 'tps-meter-code.zip').extractall(WORK)
os.chdir(WORK)
dst = pathlib.Path('/content/data/synthetic/v1'); dst.mkdir(parents=True, exist_ok=True)
for f in ['val.h5', 'test.h5', 'test_faulty.h5', 'manifest.json']:
    if not (dst / f).exists():
        shutil.copy2(DRIVE_DIR / 'data' / 'synthetic' / 'v1' / f, dst / f)
os.environ['TPS_DATA_ROOT'] = '/content/data'
os.environ['TPS_RUNS_ROOT'] = str(DRIVE_DIR / 'runs')
!pip install -q -e ".[dev]"
!python -m pytest -q tests/test_trust.py
!python scripts/evaluate.py --trust-only --ensemble-tag v1sym --seeds 1 2 3 4 5 --max-asym 10 --num-workers {os.cpu_count()}
print((DRIVE_DIR / 'reports' / 'phase6_trust' / 'TRUST_REPORT.md').read_text(encoding='utf-8'))
```
