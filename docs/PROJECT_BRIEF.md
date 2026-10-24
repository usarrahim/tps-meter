In this repository we will build the simulator, the synthetic data pipeline and the k/α estimation models (M0 and M2) of our TPS (hot-disk) thermal conductivity device. Read PROJECT_RULES.md first; its physical constants and rules apply to the whole task.

Do the work in the phases below. At the end of each phase: run the tests, write a short summary and figures under `reports/`, update `docs/PROGRESS.md`, then STOP and wait for my approval. If a validation gate fails, do not move to the next phase; investigate the cause and report it to me. Before starting, summarise these phases briefly in your own words and ask about anything unclear or contradictory.

## Folder layout

```
tps-meter/
├── docs/PROJECT_RULES.md
├── README.md
├── pyproject.toml
├── .gitignore                  # data/, runs/, .venv/, *.h5, *.pt
├── configs/
│   ├── sensor.yaml             # sensor and layer geometry (single source of truth)
│   ├── device.yaml             # power range, current ranges, sample rates, noise model
│   ├── sim_numerics.yaml       # mesh, time step, solver settings
│   ├── prior_v1.yaml           # simulation parameter ranges and distributions
│   ├── dataset_v1.yaml         # splits, sample counts, seeds
│   └── train_m2.yaml           # model and training settings
├── src/tpslib/
│   ├── config.py               # YAML → dataclass, unit checks
│   ├── dtau.py                 # analytic D(τ)
│   ├── fit_tps.py              # M0: classic TPS fit
│   ├── sim/
│   │   ├── geometry.py         # r–z mesh, layers
│   │   ├── solver.py           # finite volume, implicit time step
│   │   └── sensor_model.py     # ring sources, constant-current feedback, length-weighted readout
│   ├── measurement/
│   │   ├── acquisition.py      # clean curve → measurement at the device sample rate (noise, drift, ADC)
│   │   └── preprocess.py       # preprocessing SHARED by real and synthetic data
│   ├── data/
│   │   ├── prior.py            # parameter sampling + auto-setup rule
│   │   ├── generate.py         # parallel generation, HDF5 shards, manifest
│   │   ├── features.py         # log-time grid, θ and dθ/d(ln t)
│   │   └── dataset.py          # PyTorch Dataset, random measurement effects at training time
│   ├── models/
│   │   ├── resnet1d.py         # M2
│   │   └── losses.py           # Gaussian negative log-likelihood
│   ├── train/
│   │   └── trainer.py
│   └── eval/
│       ├── metrics.py
│       ├── calibration.py
│       └── report.py
├── scripts/                    # command-line entry points (each takes --config)
│   ├── validate_sim.py
│   ├── generate_dataset.py
│   ├── train_m2.py
│   └── evaluate.py
├── tests/
├── data/                       # not in git
│   └── synthetic/v1/           # train.h5, val.h5, test.h5, test_interp.h5, test_faulty.h5, manifest.json
├── runs/                       # not in git; each training run in its own folder
├── models/                     # released models + model cards
├── notebooks/
├── docs/                       # PROGRESS.md, DECISIONS.md
└── reports/                    # phase reports and figures
```

## Phase 0 — Skeleton and environment

- Set up the layout above, `pyproject.toml` (numpy, scipy, h5py, pyyaml, torch, matplotlib, pandas, tqdm, lightgbm, onnx, pytest) and the `.venv` environment.
- Write me one-line PowerShell commands to activate the environment and run the tests.
- Write a helper that picks CUDA automatically if available, otherwise the CPU.
- Gate: `pytest` runs without errors, even if empty.

## Phase 1 — Analytic solution and M0

- `dtau.py`: Gustafsson D(τ), parameterised by m. Write unit tests that hit the m = 20 test values in PROJECT_RULES.md within 0.05%. Add a precomputed table + interpolation for fast use and test the table error against the direct computation.
- `fit_tps.py`: fit ΔT = A·D(τ) + B, window 0.3 < τ² < 1, α scan (the α that gives the most linear result), k = P / (π^1.5·a·A).
- Gate: k and α error < 0.5% on noise-free curves generated with the analytic formula; tested for 20 different k values across 0.05–30 W/m·K.

## Phase 2 — Simulator and validation (the most critical phase)

- Axisymmetric (r–z) finite-volume model: two samples with the sensor layers between them. Model the thin layers (PI, adhesive, copper) as lumped thermal resistance + heat capacity; a separate contact conductance h_c on each face; a small convection coefficient on the outer surfaces. The mesh is refined towards the sensor plane; the time step grows logarithmically (start at 1 ms); implicit solve with sparse LU.
- Heat source: m = 20 rings, Joule heat proportional to ring length. Constant-current feedback: P(t) = I²·R0·(1 + α_R·ΔT_mean).
- Output: clean ΔT_mean(t) and P(t) on a logarithmic time grid.
- `scripts/validate_sim.py` checks these gates and writes a report:
  1. **Analytic limit:** no layers, very large h_c, constant power, very large sample: result within 0.5% of the D(τ) solution.
  2. **Convergence:** halving the mesh and the time step changes the result by less than 0.1%.
  3. **Energy conservation:** energy in and energy stored agree within 0.5%.
  4. **Physical behaviour:** apparent k drops as h_c drops; a late-time deviation starts as the sample gets thinner; the layers cause an early-time deviation. One figure each.
  5. **Speed:** a single simulation < 1 s (one core). Report the measured time and the estimated total for 60,000 simulations.
- Stop at the end of this phase without exception; we will review the figures together.

## Phase 3 — Prior, auto-setup rule and data generation

- `prior_v1.yaml` (sampled independently):
  - k: 0.05–30 W/m·K, log-uniform
  - ρc_p: 0.8–4.5 MJ/m³·K, uniform (α = k / ρc_p)
  - h_c (bottom and top separately): 500–50,000 W/m²·K, log-uniform
  - layer thicknesses: ±20% of nominal; PI and adhesive k: 0.12–0.25 W/m·K
  - sample radius 20–30 mm, thickness 8–25 mm
  - calibration error of α_R and a: ±1% and ±0.5% (normal)
- **Auto-setup rule** (`prior.py`, same logic as the device firmware): based on a "probe estimate" of k and α with ±30% error; power targeting 2.5 K at τ = 1 (clipped to 8 mW – 1 W); t_end = min(a²/α, the time at which the probing depth does not exceed 90% of the sample size); sample rate from the duration (long → 20 SPS, short → 1200 SPS). Power and duration come from this rule, they are not drawn independently.
- Generation (`generate_dataset.py`): multiprocess, with a progress bar, resumes where it stopped, writes HDF5 shards. Each sample: all parameters, clean ΔT(t), P(t), time grid. `manifest.json`: config hash, seeds, git commit, sample counts, generation time.
- First a 1,000-sample pilot: measure the time, plot histograms of the parameter distributions and of ΔT_max, show me. After my approval, the full run:

| Split | Samples | Seed | Note |
| --- | --- | --- | --- |
| train | 50,000 | 1001 | |
| val | 5,000 | 2002 | model selection |
| test | 5,000 | 3003 | only at the very end |
| test_interp | 2,000 | 4004 | from the k ∈ [1.5, 2.5] W/m·K band; in the ablation this band is removed from train/val |
| test_faulty | 2,000 | 5005 | faulty measurements: very low h_c (air gap), very thin sample, drift jump; for testing the trust system |

## Phase 4 — Measurement effects, preprocessing and features

- `acquisition.py`: sample the clean curve at the device sample rate; white + 1/f noise (level from `device.yaml`; for now a placeholder based on the ADS1263 datasheet, later replaced by a real device recording), linear + small quadratic drift (±0.2 mK/s), t_c offset, power measurement error (±0.5%), electronic drift curve (placeholder).
- `preprocess.py`: raw signal to ΔT and P; real device data go through **the same function**.
- `features.py`: resample by binning onto a 256-point logarithmic time grid; channels θ = ΔT·π^1.5·a/P and dθ/d(ln t); scalar inputs: ln t_end, ln P, ΔT_max, sample thickness and radius, current range.
- `dataset.py`: measurement effects are applied **randomly anew every epoch** (data augmentation). Normalisation statistics from train only, saved to a file.
- Gate: a figure showing 5 different noisy realisations of the same clean curve and their feature channels; the channel distributions after normalisation; a test showing that M0 still works with < 2% error on the ideal curve + noise.

## Phase 5 — M2 training

- `resnet1d.py`: 1D ResNet, 4 blocks (32–64–128–128 channels, kernel 5, GELU), global average pooling, a 128-neuron head joined with the scalars; outputs: mean and log-variance for log k, log α, log h_c. Print the parameter count (target about 200k).
- Loss: Gaussian negative log-likelihood (β-NLL, β = 0.5); target weights log k 1, log α 0.5, log h_c 0.2.
- Settings (`train_m2.yaml`): AdamW, learning rate 1e-3, cosine decay + 2-epoch warm-up, weight decay 1e-4, batch 256, at most 60 epochs, early stopping on val log k NLL (patience 8), gradient clipping 1.0, mixed precision on CUDA only.
- First a short single-seed trial (5 epochs), show the learning curves. Then a 5-seed ensemble.
- Each run under `runs/<date>_<name>_seed<n>/`: config copy, git commit, normalisation file, best weights, per-epoch metric CSV, learning-curve figure.
- Ablation: with the same settings, also train a model with the `test_interp` band removed from train/val (one seed is enough).

## Phase 6 — Evaluation and export

- `evaluate.py` reports on the test sets:
  - Median and 95th-percentile relative error for k, separately in each k range (0.05–0.5 / 0.5–5 / 5–30 W/m·K); the same for α. Target: k median < 1%, 95th percentile < 4%.
  - Error-vs-k and error-vs-h_c scatter plots.
  - Uncertainty calibration: true coverage of the 68% and 95% intervals.
  - M0 vs M2 comparison on the same test set.
  - `test_interp`: the error of the ablation model on the band it has not seen.
  - `test_faulty`: how many faulty measurements can be flagged with the ensemble uncertainty and simple checks.
- Export the best ensemble to ONNX and test that the PyTorch and ONNX outputs match. Also train a compact M2 of about 60k parameters (for the device) and report its difference to the large model.
- `models/m2_v1/MODEL_CARD.md`: training data, valid k and α range, test results, known limitations. State clearly: "These results are based on simulation only; validation with real device data has not been done yet."

Start with Phase 0 now.
