# Phase 0 report — Skeleton and environment (2026-10-17)

## What was done

- The requested folder layout was set up (root: `tps-master/`). Modules not yet written are stub files that state in which phase they will be filled.
- `pyproject.toml` (numpy, scipy, h5py, pyyaml, torch, matplotlib, pandas, tqdm, lightgbm, onnx; pytest for development), `.gitignore` (`data/`, `runs/`, `.venv/`, `*.h5`, `*.pt`, `*.onnx`).
- Config files: `sensor.yaml` with the PROJECT_RULES.md values; `prior_v1.yaml`, `dataset_v1.yaml`, `train_m2.yaml` with the values from the brief; `device.yaml` and `sim_numerics.yaml` with initial values (placeholders marked).
- `tpslib.runtime.get_device()`: picks CUDA if available, otherwise the CPU. Can be forced with `TPS_DEVICE`.
- `tpslib.config`: YAML → dataclass; unit range check that catches mm/m mix-ups.
- Colab infrastructure: `notebooks/colab_run.ipynb`, `scripts/pack_for_colab.py`, `docs/COLAB.md`.
- Environment: Python 3.11.9 (installed with `py install`), `.venv`, PyTorch with CUDA. Details below.

## Environment

| | |
|---|---|
| Python | 3.11.9 (`.venv`) |
| PyTorch | 2.11.0+cu128, CUDA 12.8 |
| `get_device()` | `cuda:0 NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.0 GB)` |
| CPU / RAM | i5-12500H, 16 threads / 15.7 GB |
| Other | numpy 2.4.6, scipy 1.17.1, h5py 3.16.0, pandas 3.0.6, matplotlib 3.11.2, lightgbm 4.7.0, onnx 1.23.2, pytest 9.1.1 |

Note: the internet connection dropped 6 times while downloading torch; the wheel was first downloaded with `pip download --resume-retries` and then installed locally. This route can be used again if the problem recurs on reinstall.

## Gate

`python -m pytest -q` → **4 passed** (configs load, sensor values match PROJECT_RULES.md, the unit guard rejects mm, `get_device()` creates a tensor on CUDA). ✔
`scripts/pack_for_colab.py` → `dist/tps-meter-code.zip` (46 files, 21 KB) built. ✔ (The Colab notebook has not yet been run on real Colab.)

No figures in this phase (there is no result to plot).

## Summary of the phases (in my own words)

1. **Phase 1 — Analytic solution and M0.** I write Gustafsson D(τ) for m rings and test it against the 6 values in PROJECT_RULES.md; table + interpolation for fast use. M0: the classic fit that scans α, finds the α at which ΔT is most linear in D(τ) and gets k from the slope. Gate: k and α error < 0.5% for 20 different k values on noise-free analytic curves.
2. **Phase 2 — Simulator.** r–z finite volume, two samples + thin layers between them (lumped resistance + capacity), a separate h_c on each face, weak convection at the outer surface, logarithmic time step, sparse LU. Source: 20 rings proportional to length, constant-current feedback, length-weighted readout. Gates: analytic limit 0.5%, convergence 0.1%, energy 0.5%, physical-behaviour figures, < 1 s/simulation. **Hard stop at the end.**
3. **Phase 3 — Prior and data.** Sample the parameters independently and derive power/duration/sample rate from the same auto-setup rule as the device firmware. Parallel, resumable HDF5 generation + manifest. First a 1,000-sample pilot and histograms, then the 5 splits after approval.
4. **Phase 4 — Measurement effects and features.** Noise/drift/ADC/power error are added to the clean curve at training time; real and synthetic data go through the same `preprocess` + `features` path. 256-point log-time grid, θ and dθ/d ln t channels + scalars. Normalisation from train only.
5. **Phase 5 — M2.** 1D ResNet, β-NLL loss, 5-seed ensemble + an ablation model with a k band removed; first a 5-epoch trial.
6. **Phase 6 — Evaluation.** Error by k range, calibration, M0–M2 comparison, interpolation and faulty-measurement tests, ONNX export, compact (~60k) model, model card.

## Open questions / contradictions

1. **h_c output.** In the prior the bottom and top h_c are sampled separately, but the model outputs a single `log h_c`. Proposal: the target is the sum of both sides `h_eff = h_bottom + h_top` (the paths from the sensor into the two samples are in parallel). Is that fine, or two separate outputs?
2. **Parameter count.** With two convolutions per block (standard ResNet), 32–64–128–128 with kernel 5 gives about **330k** parameters, which conflicts with the ~200k target. Proposal: keep the channel layout and use kernel 3 in the last two blocks (~210k). Which has priority: the architecture description or 200k?
3. **test_faulty and the "clean data" rule.** The simulator stores clean curves, but the "drift jump" is a measurement effect. Proposal: the test_faulty file holds the clean curve + a fault label, and the jump is added at evaluation time with a fixed seed through the same `acquisition` path. Is that fine?
4. **Energy-conservation gate.** There is convection loss at the outer surfaces; the balance will be tested as "energy in = stored + lost at the outer surfaces". Do I understand this correctly?
5. **M0 window.** The brief says 0.3 < τ² < 1; the common rule in the literature is 0.33 < τ < 1. Is τ² meant (i.e. about 0.55 < τ < 1), or τ?
6. **Speed gate.** < 1 s/simulation and 0.1% convergence may be hard on the same mesh. If they conflict, my priority is accuracy; for speed I use parallelism in generation (16 threads). 60,000 simulations × ~1 s / 14 processes ≈ 1–1.5 h, so local generation looks reasonable.
7. **Colab division of work.** Proposal: simulation and data generation locally (Colab Pro has few CPUs), M2 training on the Colab GPU. The local RTX 3050 Ti (4 GB) may also be enough for a model of this size; let us measure both in the 5-epoch trial of Phase 5 and decide. Do you want to move the code to Drive **as a zip package** or with **Google Drive desktop sync**?
