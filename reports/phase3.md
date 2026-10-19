# Phase 3 report — Prior, auto-setup rule and data generation: pilot (2026-10-17)

> These data are **simulation only** and consist of clean curves. Noise, drift and calibration errors are added during training in Phase 4.

## Summary

| | Result |
|---|---|
| Pilot | 1,000 samples (seed 999), **1,000 succeeded / 0 failed** |
| Energy balance (worst sample) | 7·10⁻¹¹ |
| Time | 693 s wall clock, 10 processes → **1.44 samples/s** |
| **Full-run estimate** (64,000 samples: train 50k + val 5k + test 5k + test_interp 2k + test_faulty 2k) | **~12.3 h** (on this laptop, 10 processes) |
| ΔT_max | median 2.56 K, 5–95% range 1.58–5.24 K; < 0.5 K: 0%; > 8 K: 0.6% |
| Tests | 90 passed |

Figures under `reports/phase3/`:
- `pilot_params.png`: parameter distributions.
- `pilot_autosetup.png`: power, duration, sample rate, τ_end, current range.
- `pilot_dTmax.png`: ΔT_max distribution.
- `pilot_curves.png`: example curves.
- `pilot_timing.png`: simulation times.

Numbers in `pilot_summary.json`.

## What was done

- **`data/prior.py`**
  - Independent sampling according to prior_v1. The thickness factor of each film layer is drawn separately.
  - All draws are made in a fixed order and vectorised. The same seed gives the same parameters regardless of the number of workers.
  - **Auto-setup rule** (same logic as the device firmware, plan 10.3):
    - Probe estimate: k and α each with a U(−30%, +30%) error.
    - Power: 2.5 K target at τ = 1 (with the ideal TPS formula), clipped to 8 mW–1 W.
    - Duration: `min(a²/α_probe, 2√(α_probe·t) = 0.9·min(thickness, radius − a))`.
    - Sample rate: the highest rate allowed by the budget of 12,000 samples per record (plan 7.4).
    - Current range: the L/H ranges of plan 5.3.
    - The calibrated sensor radius is used in the calculations.
  - **Model target `G_c`:** total interface conductance from the copper into the two samples (layer resistances + h_c, two paths in parallel). Bottom and top h_c are stored as well.
  - **Faults for test_faulty:** each sample has exactly one fault.
    - Air gap: h_c = 20–300 W/m²·K on one or both faces.
    - Thin sample: true thickness 2–6 mm, but the user entered the prior thickness; the auto-setup uses this wrong value.
    - Drift jump: 5–50 mK; stored only as a label and parameter, the jump is added in the Phase 4 measurement chain.
- **`data/generate.py` + `scripts/generate_dataset.py`**
  - Multiprocess generation (`spawn` for Windows) with a progress bar.
  - HDF5 shards of 250 samples are written atomically (temporary file + rename).
  - **If interrupted, the same command continues where it stopped.**
  - When a split is finished, the shards are merged into `<split>.h5`.
  - Stored per sample: all parameters, clean ΔT(t) and P(t) on a 512-point log-time grid (from 1/SPS to t_end), simulation diagnostics. A failed simulation does not stop generation; it is flagged.
  - `manifest.json` contents: all configs + sha256, seeds, git commit (no commit yet, `null`), code hash (src + configs sha256), counts and times.
- **Change in the Phase 2 simulator:** the time scheme moved to TR-BDF2 (rationale below). The Phase 2 gates were re-run with the new scheme and all passed; `reports/phase2.md` was updated.
- **Bug fix:** PyYAML read unsigned exponent numbers such as `0.8e6` as text. The loader was fixed; a test was added that checks no numeric text remains in the configs. Earlier results were not affected, because the code already converted these values with `float()`.

## Parallel generation time: what was tried

| Change | Throughput (samples/s) | 64k estimate |
|---|---|---|
| Phase 2 scheme (Richardson-Euler, 16 steps), 14 processes | 1.15 | ~15.5 h |
| Same scheme, 8 processes (1–16 scanned) | 0.96 | — |
| **TR-BDF2, 12 steps, 6–12 processes** | **1.44–1.59** | **~11–12 h** |

- A simulation that takes 2.3 s alone takes 6.9 s while 10 processes run in parallel. The bottleneck is not the CPU but **memory bandwidth**: the sparse LU back-substitutions read large triangular matrices from memory.
- TR-BDF2 halved the number of solves per step; the accuracy is the same (Phase 2 gates passed again).
- Throughput does not increase beyond 8 processes.

## Pilot findings

1. **Distributions as expected.** k log-uniform, ρc_p uniform; α 0.013–20 mm²/s. G_c 1,000–16,000 W/m²·K, middle ~6,000.
2. **Auto-setup.**
   - In 21% of the samples the power clips at 1 W (k ≳ 7 W/m·K), in 2% at 8 mW.
   - In 43% of the measurements the duration is limited by the sample size (probing depth) rather than τ = 1.
   - True τ_end 0.52–1.21; because of the probe error some measurements exceed τ = 1.
   - Sample rates: 20 SPS 31%, 50 SPS 12%, 100 SPS 23%, 400 SPS 16%, 1200 SPS 19%.
3. **The film/contact offset increases ΔT_max.** The rule targets 2.5 K with the ideal formula, but the sensor's own thermal resistance and h_c add an offset.
   - At high k and low G_c ΔT_max reaches 6–10 K; 0.6% exceed the 8 K limit of the trust system.
   - **Open question:** if the real firmware chooses the power from the ΔT *measured* during the probe pulse, this offset is compensated automatically. I implemented the "probe k + ideal formula" rule requested in the brief. Whatever rule the firmware uses, the simulator should use the same.
4. **High k will be the hard region.** For the k = 30 example most of the normalised curve comes from the film offset; the late-time slope that carries k is small (`pilot_curves.png`, right panel). I expect a large error in the 5–30 W/m·K band in Phase 6. The plan also gives the validated range as 0.19–16 W/m·K.

## Options for the full run

1. **All on this computer:** ~12 h, 10 processes, one night. Command:
   `python scripts/generate_dataset.py --splits train val test test_interp test_faulty --workers 10`
   If interrupted, the same command continues.
2. **Split:** train (50k, ~9.5 h) on this computer; val, test, test_interp and test_faulty (14k) at the same time on Colab. The total time drops to ~9.5 h. The splits have separate seeds and are deterministic, so the result is the same; the two manifests are merged afterwards. The speed of the Colab CPU runtime has not been measured yet.

**I will not start the full run without your approval.**
