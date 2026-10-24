# tps-meter — AI-assisted hot-disk (TPS) thermal conductivity measurement

ITU EHB graduation project: a low-cost, USB-powered, AI-assisted **hot-disk (Transient Plane Source)** thermal conductivity meter.
This repository holds the device's **analysis software**:
- a physics simulator of the sensor + sample system,
- a synthetic data pipeline,
- the classic TPS solution (M0),
- a neural network (M2) that estimates thermal conductivity (k), thermal diffusivity (α) and contact conductance, each with an uncertainty.

> **Confidentiality.** This is a private repository of proprietary software; see [LICENSE](LICENSE). It includes the trained model weights (`runs/*/best.pt`, `last.pt`), the ONNX exports and normalisers (`models/m2_v1/`) and the small pilot data. The full synthetic dataset (several GB) stays on Google Drive and can be regenerated from the seeds in `configs/dataset_v1.yaml`.

## Status

| Stage | Status | Report |
|---|---|---|
| Phases 0–1: infrastructure, analytic D(τ), classic TPS (M0) | done | [phase0](reports/phase0.md), [phase1](reports/phase1.md) |
| Phase 2: r–z finite-volume simulator and validation | done | [phase2](reports/phase2.md) |
| Phase 3: prior, auto-setup rule, 64,000 synthetic curves | done | [phase3](reports/phase3.md) |
| Phase 4: measurement chain, shared preprocessing, features | done | [phase4](reports/phase4.md) |
| Phases 5–6: M2 ensemble, test evaluation, ONNX | done (base model M2 v1) | [phase5](reports/phase5.md), [phase6](reports/phase6.md), [model card](models/m2_v1/MODEL_CARD.md) |
| M0+ re-simulation inversion; base model closed (`v1.0-base`) | done | [m0plus](reports/m0plus.md) |
| Calibration and validation with real device data | next | — |

Full write-up of the work so far: [reports/REPORT_v1.md](reports/REPORT_v1.md).

**M2 v1 test result** (simulation only; k median / p95 error):

| k range | M2 v1 | M0 (classic) |
|---|---|---|
| 0.05–0.5 W/m·K | 0.83% / 2.6% | 1.21% / 3.7% |
| 0.5–5 W/m·K | 0.89% / 3.2% | 1.41% / 6.4% |
| 5–30 W/m·K | 2.07% / 10.4% | 5.03% / 23.9% |

Accuracy on the real device has not been measured yet.

## Repository layout

```
configs/        all settings (SI units): sensor, device, simulation numerics, prior, dataset, training
src/tpslib/
  dtau.py, fit_tps.py        analytic D(τ) and the classic TPS fit (M0)
  sim/                       r–z finite-volume simulator (mesh, solver, sensor model, analytic reference)
  measurement/               measurement-chain model and the preprocessing SHARED by real and synthetic data
  data/                      prior + auto-setup, parallel data generation, features, PyTorch dataset
  models/                    1D ResNet (M2), β-NLL loss, ONNX export
  train/                     training loop (resumable)
  eval/                      metrics, prediction, batch M0, trust checks, plot style
scripts/        command-line entry points (validation, generation, training, diagnosis, evaluation)
tests/          pytest (every physics function has a test)
notebooks/      Google Colab runner
reports/        phase reports and figures
docs/           progress log, design decisions, Colab guide, project brief
models/         released models: ONNX exports, normalisers, trust thresholds, model cards
runs/           every training run: PyTorch weights (best.pt, last.pt with full training history), normaliser, curves
```

## Setup (Windows, PowerShell)

Requires Python 3.11.

```powershell
py -V:3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Activate the environment and run the tests:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## Common commands

| Task | Command |
|---|---|
| Simulator validation gates | `python scripts/validate_sim.py` |
| Data generation (resumable) | `python scripts/generate_dataset.py --splits train val test test_interp test_faulty` |
| M2 training (5 seeds) | `python scripts/train_m2.py --seeds 1 2 3 4 5 --tag v1sym --max-asym 10` |
| Error diagnosis (val) | `python scripts/diagnose.py --run v1sym_m2_seed1` |
| Final evaluation + ONNX | `python scripts/evaluate.py --ensemble-tag v1sym ...` |

Long jobs run on Google Colab: [docs/COLAB.md](docs/COLAB.md), [notebooks/colab_run.ipynb](notebooks/colab_run.ipynb). Data and training outputs are located through the `TPS_DATA_ROOT` and `TPS_RUNS_ROOT` environment variables.

## Documents

- [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md): phase-by-phase work definition
- [docs/PROGRESS.md](docs/PROGRESS.md): progress log
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions and their rationale
- [docs/PROJECT_RULES.md](docs/PROJECT_RULES.md): project rules and physical constants (single source of truth: `configs/sensor.yaml`)

## License

Proprietary; all rights reserved. See [LICENSE](LICENSE).
