# M2 v1 — model card

> **These results are based on simulation only; validation with real device data has not been done yet.**
> Real accuracy will only be established with reference materials, material-level cross-validation (plan T8) and locked test materials (T9).

| | |
|---|---|
| Version | M2 v1 (base model, trained on simulation only) |
| Date | 2026-10-19 |
| Ensemble | 5 members (`v1sym_m2_seed1…5`), 375,078 parameters per member |
| Compact device model | 2 members (`v1c_m2c_seed1…2`), 61,910 parameters per member |
| Code | git `6d03bf1` and later; data v1 manifest `data/synthetic/v1/manifest.json` |

## What it does

From a TPS (hot-disk) measurement curve it estimates three quantities, each with its uncertainty:
- **k:** thermal conductivity
- **α:** thermal diffusivity
- **G_c:** total interface conductance from the sensor into the two samples

The uncertainty is the sum of two parts: the per-model uncertainty and the spread between the 5 members.

**Usage chain** (the same for real and synthetic data):
1. Raw device record (ADC ratio, current, R₀, drift) → `tpslib.measurement.preprocess`
2. → `tpslib.data.features.make_features`: 3 channels × 256 log-time bins + 7 scalars
3. → `m2_ensemble.onnx`: normalisation is inside the graph. Outputs in ln units: `mean`, `std` and the member estimates `member_mean`.

```python
import numpy as np, onnxruntime as ort
from tpslib.measurement.preprocess import preprocess
from tpslib.data.features import make_features
f = make_features(preprocess(raw_record))
sess = ort.InferenceSession("models/m2_v1/m2_ensemble.onnx")
mean, std, members = sess.run(None, {"seq": f["seq"][None], "scalars": f["scalars"][None]})
k, k_lo, k_hi = np.exp(mean[0, 0]), np.exp(mean[0, 0] - 1.96 * std[0, 0]), np.exp(mean[0, 0] + 1.96 * std[0, 0])
```

The largest difference between ONNX and PyTorch is 2·10⁻⁶ (ln units).

## Training data

- **Source:** synthetic data v1, 64,000 simulations, 0 failures.
- **Simulator:** r–z finite volume. 20-ring copper spiral (a = 6.4 mm). PI / copper+adhesive / adhesive / PI film layers. Separate contact conductance on each face. Constant-current feedback. Validation gates: `reports/phase2.md`.
- **Measurement parameters:** power, duration and sample rate come from the device's auto-setup rule.
- **Training set:** 37,604 samples with contact asymmetry (h_c,max/h_c,min) ≤ 10.
- **Measurement effects:** re-applied every epoch. Noise, drift, t_c, power error, electronic drift, α_R and radius calibration errors (`configs/device.yaml`, currently placeholder values).

## Valid range

| Parameter | Range |
|---|---|
| k | 0.05–30 W/m·K |
| ρc_p | 0.8–4.5 MJ/m³·K (α ≈ 0.011–37 mm²/s) |
| Sample | thickness 8–25 mm, radius 20–30 mm, two identical pieces |
| Contact | h_c 500–50,000 W/m²·K on each face, at most a 10× difference between the faces |
| Measurement | power 8 mW–1 W, sample rate 20–1200 SPS, τ_end ≈ 0.5–1.2 |

## Test results (locked test set, realistic contact domain, n = 3,715)

| k band [W/m·K] | n | M2 k median | M2 k p95 | M0 k median | M0 k p95 | M2 α median | M2 α p95 |
|---|---|---|---|---|---|---|---|
| 0.05–0.5 | 1,392 | **0.83%** | **2.60%** | 1.21% | 3.65% | 0.71% | 2.47% |
| 0.5–5 | 1,346 | **0.89%** | **3.17%** | 1.41% | 6.39% | 0.92% | 6.81% |
| 5–30 | 977 | 2.07% | 10.36% | 5.03% | 23.86% | 2.78% | 9.32% |
| All | 3,715 | 1.01% | 5.85% | 1.71% | 13.47% | 1.03% | 7.32% |

- **The simulation target** (k median < 1%, p95 < 4%) **is met for k < 5 W/m·K.** It is not met in the 5–30 band.
- **Irreducible floor:** calibration errors (α_R σ 1%, radius σ 0.5%) and the ±0.5% power error cannot be seen in the curve. These alone cause ~0.77% median and ~2.3% p95 error in k.
- **Uncertainty calibration:** the 95% interval contains the true value 97.0% of the time for ln k, 99.3% for ln α and 96.4% for ln G_c. The ln α uncertainty is wider than needed (conservative).
- **Interpolation:** the ablation model that never saw the k = 1.5–2.5 band gave 0.92% / 2.91% error on that band. The model that saw the band: 0.86% / 2.58%. Generalisation to intermediate values is good.
- **Compact model** (2 × 61,910 parameters): k 1.08% / 6.14% over all. Difference to the large ensemble: median 0.27%, p99 2.1%.

## Trust checks (v1)

Thresholds were set from the 99th percentiles of the validation set. Checks used: ensemble spread, total uncertainty, M0–M2 difference, M0 residual and ΔT range.

| Case | Flagged |
|---|---|
| Thin sample (2–6 mm, wrongly entered thickness) | 97.9% |
| Drift jump (5–50 mK) | 84.9% |
| Air gap (h_c 20–300) | **52.8%** |
| Contact asymmetry > 10 (outside the training domain) | **4.2%** |
| Normal measurement (false alarm) | 4.7% |

## Trust checks (v2) and uncertainty scale

The model was not retrained; only new checks were added on its outputs (`tpslib.eval.trust`, settings in `configs/trust.yaml`). All thresholds and scales are chosen **on val only** (realistic contact domain) and written to `trust_v2.json`.

- **Training-range check (plan 9.8):** flagged if the predicted ln G_c, ln k or ln α lies outside the range the model produces on normal val measurements (both ends).
- **Ensemble spread for every target:** v1 only looked at the member spread of ln k; v2 also looks at ln α and ln G_c.
- **Shared false-alarm budget:** so that false alarms do not grow with the number of checks, all checks are tuned with a single tail share; the total false alarm on val stays below 5%.
- **Variance scale:** the std of each target is multiplied by a factor found on val, bringing the 50%, 68%, 90% and 95% intervals to their nominal coverage. In Phase 6 the ln α 68% interval covered 89% (wider than needed). The ONNX output is unscaled; the factor is in `trust_v2.json` and `std × scale` is applied on the user side.

| Metric | Value |
|---|---|
| Variance scale ln k / ln α / ln G_c | to be filled in after the Colab re-run |
| Flagged: air gap / thin sample / drift jump / asymmetry > 10 | to be filled in after the Colab re-run |
| False alarm on normal measurements | to be filled in after the Colab re-run |

These results are the **second use** of the test sets (the first was Phase 6). The thresholds were not chosen by looking at the test set. The re-simulation check (M0+) does not exist yet; its design is in `docs/M0PLUS.md`.

## Known limitations

1. **Simulation only.** Noise, drift, t_c, electronic drift and the real contact distribution of the real device were modelled with placeholder values. When real data arrive, `device.yaml` will be updated and the model retrained, with data v2 generated if needed (plan 9.7).
2. **High k (> 5 W/m·K) is the hardest region.** Power clips at 1 W and most of the measured temperature rise comes from the sensor's own layers and the contact. The p95 error in this band is ~10%.
3. **Poor and asymmetric contact is not caught well enough.** Half of the air-gap measurements and most strongly asymmetric contacts are not flagged. In unflagged faulty measurements k comes out ~14% low. Two checks of plan 9.8 are still missing:
   - whether the predicted G_c lies within the training range
   - the re-simulation residual with the estimated parameters (M0+)

   Until these are added, a low k result should not be used without a contact check.
4. **The α uncertainty is conservative.** The observed coverage is higher than expected. The val-fitted scale factor in trust v2 corrects this (above).
5. **Physics not modelled:** lead heating, PETG fixture plate, anisotropy, temperature-dependent material properties, surface roughness, liquids and porous materials.
6. **Single sensor geometry:** variant A (a = 6.4 mm, m = 20, b = 0.10 mm). Variants B/C or a different film stack need new data generation and training.

## Files

| File | Description | Git |
|---|---|---|
| `m2_ensemble.onnx` | 5-member ensemble + normalisation (7.6 MB) | no (Drive: `tps-meter/phase6_bundle.zip`) |
| `m2c_compact.onnx` | 2-member compact device model (0.5 MB) | no |
| `normalizer_v1sym_m2_seed*.json` | Member normalisations (from train only) | no (model file, confidential) |
| `trust_v2.json` | Trust v2 thresholds and variance scale (from val) | no (Drive: `tps-meter/phase6_trust_bundle.zip`) |
| PyTorch weights | Drive: `tps-meter/runs/v1sym_m2_seed*/best.pt`, `v1c_m2c_seed*/best.pt` | no |
