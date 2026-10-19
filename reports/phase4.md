# Phase 4 report — Measurement effects, preprocessing and features (2026-10-18)

> The measurement-chain model runs on **placeholder** values: the ADS1263/1262 datasheet and the project plan. When an idle recording of the real device (plan T2/T3) arrives, `device.yaml` will be updated. **The data do not need to be regenerated** for this, because the effects are added during training. The results in this phase are based on simulation.

## Gate summary

| Gate | Result | Status |
|---|---|---|
| 5 noisy realisations of the same clean curve and the feature channels | `phase4/realisations_and_channels.png` (low and high k) | ✔ |
| Distributions after normalisation | all 3 channels and 7 scalars have mean 0, std 1, finite (`normalised_distributions.png`) | ✔ |
| M0, ideal curve + measurement effects, < 2% | 20 k × 5 repeats: largest k error **0.70%**; **1.59%** with calibration errors added | ✔ |
| Tests | **125 passed** (34 new Phase 4 tests) | ✔ |

Normalisation was tried here on the pilot data. The real statistics are computed in Phase 5 **on train only** and saved with the model. The code refuses to compute statistics from a split other than train; this is tested too.

## What was done

### `measurement/acquisition.py`: clean curve → raw device record

The synthetic output has the same fields the real device records:
- ADC1 ratio R/R_ref (32-bit quantisation)
- ADC2 current (10 SPS)
- R₀ and drift slope from the pre-measurement
- Calibrated α_R and radius

Effects added (`device.yaml`, random at every call):

| Effect | Model |
|---|---|
| Noise | White: 0.23 µV × √(SPS/20) × (1–2×, plan T2 limit). Temperature equivalent V/(I·R₀·α_R); 0.15 mK at 100 mA and 20 SPS, same as the plan. 1/f corner 0–1 Hz. |
| Ambient drift | ±0.2 mK/s linear + small quadratic. 5 µK/s error in the pre-measurement slope estimate. Preprocessing can only remove the estimated linear part. |
| Timing | Heating start offset t_c ±5 ms, clock error 20 ppm |
| Power | ±0.5% absolute current/power error, ADC2 noise |
| Calibration | α_R (σ 1%) and radius (σ 0.5%) from the prior; R₀ measurement error; R_ref tolerance (cancels in the ratiometric measurement, tested) |
| Electronic drift | R_ref self-heating 10 mK/W, τ 20 s; correction curve with ±50% error |
| Fault | drift jump for test_faulty |

### `measurement/preprocess.py`: shared preprocessing (plan 9.2)

- Steps: R = ratio × R_ref → ΔT = (R/R₀ − 1)/α_R → subtract the pre-measurement drift → subtract the electronic drift curve → P = I²·R.
- **Real device data will go through the same function.**
- With the effects switched off, the chain returns the clean curve within 10⁻⁷ K (tested).

### `data/features.py`

- **Grid:** 256 log-time bins on [t_end/1000, t_end]. In sparse bins the values are moved from the bin's mean ln t position to the bin centre by interpolation.
- **Channels** (the third is my addition):
  - `θ/θ_ref`
  - `dθ/d ln t / θ_ref` (Savitzky–Golay, 15 bins)
  - `ln(1 + samples per bin)`
  - Rationale below.
- **Scalars:** ln θ_ref, ln t_end, ln P₀, ΔT_max, entered thickness and radius, current range.
- **Targets:** ln k, ln α, ln G_c.

### `data/dataset.py`

- PyTorch Dataset that reads clean curves from HDF5.
- **In train the effects are redrawn every epoch** (seed = base, epoch, sample). In val and test they are fixed per sample, so the metrics are repeatable.
- HDF5 is opened separately in each worker process (Windows compatible).
- k-band exclusion for the ablation (`select_indices`).
- `Normalizer`: computed from train only, saved as JSON with the model.
- **Speed:** 4.2 ms per sample. An epoch of 50k samples takes 3.5 min on one core, about 0.5 min with 8 workers.

## Design decisions and rationale

1. **Divide θ by θ_ref and make ln θ_ref a scalar.** θ ≈ D(τ)/k + const, so it varies 600× over 0.05–30 W/m·K. Given raw, the network would miss the shape while struggling with the magnitude. With this split **no information is lost**: the amplitude is in ln θ_ref, the shape in the channel. That is the difference from ΔT/ΔT_max, where the amplitude and hence the k information were lost.
2. **Sample-count channel (beyond the brief).** In long measurements at 20 SPS the early bins are empty or hold one sample, the late bins hold hundreds. The channel tells the model how noisy each bin is. The information comes entirely from the measurement layout; there is no leakage.
3. **The pre-measurement estimates the drift slope with an error, and the quadratic drift cannot be removed.** This is also true in reality; the model must learn to be robust to this residual.

## Sensitivity of M0 to the measurement effects

Each effect alone, worst of 20 repeats (`phase4/m0_effect_sensitivity.csv`):

| Effect | k=0.05 (820 s, 20 SPS) | k=0.19 (350 s, 20 SPS) | k=1.2 (85 s, 100 SPS) | k=16 (9 s, 1200 SPS) |
|---|---|---|---|---|
| **Ambient drift** | k 0.51% / **α 10.3%** | k 0.24% / **α 4.9%** | k 0.06% / α 1.2% | k 0.02% / α 0.3% |
| Noise | k 0.31% / α 1.8% | k 0.21% / α 1.8% | k 0.07% / α 0.4% | k 0.12% / α 0.8% |
| Power error | k 0.46% / α 0.0% | k 0.46% / α 0.0% | k 0.46% / α 0.0% | k 0.46% / α 0.0% |
| Calibration (α_R 1%, a 0.5%) | k 0.58% / α 1.0% | same | same | same |
| Electronic drift | ~0 | ~0 | k 0.06% / α 0.2% | k 0.23% / α 1.7% |
| t_c, clock | ~0 | ~0 | ~0 | k 0.16% / α 0.3% |

Interpretation:
- **k is robust:** every effect stays below 0.6%; consistent with the uncertainty budget in the plan (10.2).
- **For long, low-k measurements α is dominated by the ambient drift (5–10%).** Consistent with the plan's warning "ambient drift 1–2% for PMMA"; α is much more sensitive to the curve shape than k. M2 sees the drift in training, so it should do better than M0 here; we will measure it in Phase 6.
- **No method can correct calibration errors.** 1% in α_R goes straight into k (partly cancelled by the radius). The α_R calibration (plan T4) is therefore critical.

## Open points (to be settled with the real device)

- **t_c (heating start offset) ±5 ms is a placeholder.** At high k (1200 SPS) it changes the first ~30 ms noticeably (`realisations_and_channels.png`, bottom row). If the DAC command and the DRDY timestamp come from the same timer, the real value is much smaller. To be measured once the firmware is done.
- Noise level, 1/f corner, drift statistics and the electronic drift curve will be taken from idle and dummy-load recordings of the device (plan 9.7).
- **Important:** `configs/device.yaml` changed. **Do not upload a new zip** before the Colab generation finishes. The changed config would make it reject the unfinished shards as "different code".
