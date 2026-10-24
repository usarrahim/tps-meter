# Project rules — tps-meter (ML and simulation part)

These rules apply to every change in this repository.

## Project

ITU EHB graduation project: a low-cost, AI-assisted **hot-disk (Transient Plane Source, TPS)** thermal conductivity meter.
The work in this repository: **write the simulator of the sensor + sample system, generate synthetic data and train the model that estimates k (thermal conductivity) and α (thermal diffusivity).**
Detailed plan: project handbook, Chapter 9 (model training handbook). If they conflict, these rules win; record the conflict in `docs/DECISIONS.md` and ask the user.

## Physics and constants (single source of truth: `configs/sensor.yaml`)

- Sensor: copper double spiral, m = 20 concentric rings, outer radius a = 6.4 mm, trace width b = 0.10 mm, ring radii r_i = i·a/m.
- Copper thickness 18 µm. Bottom side: 25 µm polyimide base. Top side: 12.5 µm polyimide + 15 µm adhesive. **The two sides are not symmetric.**
- R0 ≈ 4 Ω (measured), temperature coefficient of resistance α_R ≈ 0.0039 1/K (calibrated).
- Measurement at constant current: P(t) = I²·R0·(1 + α_R·ΔT(t)). The power is computed together with the temperature.
- The temperature the sensor reads = the **length-weighted mean** of the ring temperatures (this is what the resistance measurement itself gives).
- Samples: two identical pieces, default radius 25 mm, thickness 20 mm (range in the prior).
- Device power range: 8 mW – 1 W. Sample rate: 20–1200 samples/s. Measurement time chosen so that τ ≤ 1, τ = sqrt(α·t)/a.
- Target temperature rise: about 2–3 K at τ = 1.
- Analytic reference: Gustafsson D(τ). Test values for m = 20:
  D(0.1)=0.20553, D(0.3)=0.35460, D(0.5)=0.46487, D(0.75)=0.55840, D(1.0)=0.61900, D(1.5)=0.68912
  (use `scipy.special.i0e` to avoid overflow in the product exp(-(l²+k²)/(4m²σ²))·I0(lk/(2m²σ²)).)
- Ideal case: ΔT_mean(τ) = P / (π^1.5 · a · k) · D(τ).

## Technical rules

- Python 3.11. Environment: `.venv` (venv) in the project root. Dependencies in `pyproject.toml`. Package code under `src/tpslib/`.
- **The operating system is Windows.** Use `pathlib` for paths. Multiprocess code must have an `if __name__ == "__main__":` guard (spawn on Windows). Write commands that run in PowerShell; no bash-specific syntax.
- Units inside the code are **SI** (m, s, W, K). Interfaces and reports may show mm, mW; conversion happens only at input/output.
- Every setting lives in `configs/*.yaml`. No magic numbers in code; constants are read from the configs.
- Randomness: every generation and training run uses a seed, and the seed is written to the output folder. Same config + same seed = same result.
- Data and model weights do not go into git (`data/`, `runs/`, large files are in `.gitignore`). Into git go: code, configs, manifests, model cards, reports.
- Tests with `pytest`. Every physics function has a test. Code that fails its tests does not move to the next stage.
- **Language:** everything in the repository is English — code comments, docstrings, variable names, plot labels, reports, model cards and docs. Only the chat with the user is in Turkish.

## Data rules (these prevent leakage and fake success)

- The simulator produces and stores **clean** curves. Noise, drift, sample rate and ADC effects are added **during training**, through **the same preprocessing** that real device data will go through. Real and synthetic data pass through the same `features` function.
- Splits are generated separately with separate seeds: train / val / test never share a parameter sample.
- Normalisation statistics are computed **on train only** and saved with the model.
- The test set is not used for model selection. Selection is done on val; test is reported once at the end.
- Success on the simulation test is not real-world success. Reports always state this distinction.

## Way of working

- Work is done in phases. At the end of each phase: tests run, a short English summary + figures are written under `reports/`, `docs/PROGRESS.md` is updated and **the user's approval is awaited.**
- For long jobs (data generation, training) first run a small pilot, measure the time, estimate the full run and tell the user.
- If a validation gate fails, do not move on; investigate and report the cause.
- Record important design decisions with their rationale in `docs/DECISIONS.md`.
