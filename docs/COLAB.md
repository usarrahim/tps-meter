# Working with Google Colab

All long jobs (data generation, training) run on Colab. Code is developed on the computer.

## How the code is moved: Drive (zip)

- **Chosen route: zip package via Drive.** Data and training outputs live on Drive anyway. No extra account, token or repository setup is needed.
- **When does GitHub make sense?** Once you start developing together with a teammate (plan 9.10). If the repository is private, Colab needs a personal access token. The code then comes in with `git clone`; the rest of the flow stays the same.

## Setup (once)

1. Create the folder `My Drive/tps-meter/` on Drive.
2. Build the code package on the computer:
   `python scripts/pack_for_colab.py` → `dist/tps-meter-code.zip`
3. Drag and drop this zip into the `tps-meter/` folder on Drive. Every time the code changes, rebuild the package and upload it again, replacing the old file.
4. Open `notebooks/colab_run.ipynb` in Colab. The notebook is also inside the zip; you can also put it on Drive, double-click it and choose "Open with Google Colaboratory".

## Choosing the runtime (data generation)

- Generation uses the CPU only and is limited by **memory bandwidth**. On this laptop the speed-up stopped after 6 processes.
- Choose the runtime that gives the most CPU cores. How many cores each type gives may depend on the account; cell 2 prints the core count.
- Cell 4 (`bench_generation.py`) measures a few worker counts and prints the fastest one and the estimated time of the full run. Set `WORKERS` in cell 5 accordingly.
- The GPU is not used in this step. A GPU runtime only helps if it gives more CPU cores.

## If the session disconnects

- Shards (250 samples) are generated on **Colab's local disk** and copied to `My Drive/tps-meter/data/synthetic/v1/` as soon as they finish (`--mirror`).
- If the session drops, at most the shards in progress are lost (a few minutes of work).
- **To continue:** run cells 1, 2 and 5 again. Finished shards are restored from Drive and the rest are generated.
- If the code changed, old shards are **rejected** (code_hash check). This prevents different simulator versions from mixing in one dataset.
- Keep the browser tab open. If the computer goes to sleep, Colab may drop the connection. Cell 6 shows the completion status.

## Splits

| Split | Samples | Seed |
|---|---|---|
| train | 50,000 | 1001 |
| val | 5,000 | 2002 |
| test | 5,000 | 3003 |
| test_interp | 2,000 | 4004 |
| test_faulty | 2,000 | 5005 |

The same config, code and seed give the same parameters and the same clean curves on every machine. The number of parallel workers does not change the result.

## Training (Phase 5)

The data are copied from Drive to Colab's local disk and read from there (reading HDF5 directly from Drive is slow). Training outputs are written directly to Drive (`TPS_RUNS_ROOT`).

## Phase 6 evaluation (English figures) + trust checks v2

1. On the computer, `python scripts/pack_for_colab.py` → `dist/tps-meter-code.zip`; drop it into the `tps-meter/` folder on Drive, replacing the old one.
2. Open a new notebook in Colab (the Phase 6 runtime is enough), paste the cell below and run it. The last cell of the notebook is the same.
3. At the end `TRUST_REPORT.md` is printed; send this text back. Files: `tps-meter/reports/phase6/` and `tps-meter/reports/phase6_trust/`.

The full evaluation repeats the Phase 6 numbers (same models, same seeds) with English figures. The trust run is the **second use** of the test sets (the first was Phase 6). Thresholds are chosen on val.

```python
# 15) Phase 6 evaluation (English figures) + trust checks v2 and variance scale (SECOND USE OF THE TEST SETS).
# Runs on its own; ~30-40 min.
from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, shutil, zipfile
DRIVE_DIR = pathlib.Path('/content/drive/MyDrive/tps-meter')
WORK = pathlib.Path('/content/tps-meter')
shutil.rmtree(WORK, ignore_errors=True)
zipfile.ZipFile(DRIVE_DIR / 'tps-meter-code.zip').extractall(WORK)
os.chdir(WORK)
dst = pathlib.Path('/content/data/synthetic/v1'); dst.mkdir(parents=True, exist_ok=True)
for f in ['val.h5', 'test.h5', 'test_interp.h5', 'test_faulty.h5', 'manifest.json']:
    if not (dst / f).exists():
        shutil.copy2(DRIVE_DIR / 'data' / 'synthetic' / 'v1' / f, dst / f)
os.environ['TPS_DATA_ROOT'] = '/content/data'
os.environ['TPS_RUNS_ROOT'] = str(DRIVE_DIR / 'runs')
!pip install -q -e ".[dev]"
!python -m pytest -q tests/test_trust.py tests/test_phase6.py
!python scripts/evaluate.py --ensemble-tag v1sym --seeds 1 2 3 4 5 --ablation ablation_m2_seed1 --compact-tag v1c --compact-seeds 1 2 --max-asym 10 --num-workers {os.cpu_count()}
!python scripts/evaluate.py --trust-only --ensemble-tag v1sym --seeds 1 2 3 4 5 --max-asym 10 --num-workers {os.cpu_count()}
print((DRIVE_DIR / 'reports' / 'phase6_trust' / 'TRUST_REPORT.md').read_text(encoding='utf-8'))
```
