# tps-meter — simülatör, sentetik veri ve k/α modeli

İTÜ EHB bitirme projesi: düşük maliyetli, yapay zekâ destekli hot-disk (TPS) ısıl iletkenlik cihazı.
Bu depo: sensör + numune simülatörü, sentetik veri üretimi, M0 (klasik TPS uydurması) ve M2 (1B ResNet) modelleri.
Kurallar ve fizik sabitleri: [docs/PROJECT_RULES.md](docs/PROJECT_RULES.md). İlerleme: [docs/PROGRESS.md](docs/PROGRESS.md). Kararlar: [docs/DECISIONS.md](docs/DECISIONS.md).

## Yerel ortam (Windows, PowerShell)

Proje klasöründe:

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pytest -q
```

İlk kurulum (zaten yapıldı; yeniden kurmak gerekirse):

```powershell
py -V:3.11 -m venv .venv; .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`Activate.ps1` "running scripts is disabled" hatası verirse bir kez:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Google Colab

Ayrıntılar: [docs/COLAB.md](docs/COLAB.md). Kısaca: `python scripts/pack_for_colab.py` → `dist/tps-meter-code.zip` dosyasını
Drive'da `MyDrive/tps-meter/` klasörüne koy → [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb) dosyasını Colab'da aç.

## Klasörler

- `configs/` tüm ayarlar (SI birimleri) · `src/tpslib/` paket kodu · `scripts/` komut satırı girişleri · `tests/` pytest
- `data/`, `runs/` git'e girmez · `models/` yayınlanan modeller ve model kartları · `reports/` faz raporları
