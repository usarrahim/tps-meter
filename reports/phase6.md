# Phase 6 report — Evaluation and export (2026-10-19)

> **These results are based on simulation only; validation with real device data has not been done yet.** Success on the simulation test is not real-world success. The project success criterion "≤ 15% on real reference materials" is separate and will be measured in a later stage.

The test sets (`test`, `test_interp`, `test_faulty`) were used here **once**. Model selection and all thresholds were done on the validation set. Model card: [`models/m2_v1/MODEL_CARD.md`](../models/m2_v1/MODEL_CARD.md). Raw outputs: `reports/phase6/` (`summary.json`, CSVs, figures).

## Gate summary

| Item (brief) | Result | Status |
|---|---|---|
| k median < 1%, p95 < 4%, in every k band | 0.05–0.5: 0.83% / 2.60% · 0.5–5: 0.89% / 3.17% · 5–30: 2.07% / 10.36% | ✔ ✔ ✘ |
| α in the same bands | 0.71% / 2.47% · 0.92% / 6.81% · 2.78% / 9.32% | reported |
| Error-vs-k and error-vs-h_c plots | `test_error_scatter.png` | ✔ |
| Coverage of the 68% / 95% intervals | ln k: 74.7% / 97.0% · ln α: 89.3% / 99.3% · ln G_c: 69.1% / 96.4% | ✔ (k and α conservative) |
| M0–M2 comparison | All, k: M2 1.01% / 5.85% · M0 1.71% / 13.47% | ✔ M2 clearly better |
| test_interp (ablation) | Model that did not see the band 0.92% / 2.91%; model that saw it 0.86% / 2.58% | ✔ |
| test_faulty flagging (plan target 90%) | Thin sample 97.9% · drift jump 84.9% · air gap 52.8% | ✘ partly |
| ONNX identical to PyTorch | Largest difference 2·10⁻⁶ | ✔ |
| Compact model (~60k) | 2 × 61,910 parameters: k 1.08% / 6.14%; difference to the ensemble median 0.27% | ✔ |
| Model card | `models/m2_v1/MODEL_CARD.md` | ✔ |

## Interpretation

1. **In the main target range (k < 5 W/m·K) the simulation target is met.** The project's main reference set (PMMA 0.19, PTFE 0.25, borosilicate 1.2) lies in this range.
   - The error is very close to the irreducible floor from the calibration errors (~0.77% median).
   - The model itself is not the bottleneck in this band. The real accuracy will be set by the α_R calibration, the power measurement, the contact and the uncertainty of the reference values themselves.
2. **The high-k (5–30 W/m·K) target is not met.** p95 error 10%, median 2%.
   - A physical difficulty known since Phase 2: power clips at 1 W, and most of the signal comes from the sensor's own layers.
   - Still 2–2.5× better than the classic method. M0's systematic −5% bias in this band is absent in M2 (`test_error_scatter.png`, middle panel).
   - The validated range in the plan is 0.19–16 W/m·K anyway. Stainless steel (16) lies inside this band.
3. **M0's systematic error is clear.** +2–3% at low k, −5% to −30% at high k. This bias comes from the sensor layers and the contact; it should be expected on the real device too. This is what makes M2 and the planned M0+ (re-simulation inversion) valuable.
4. **Generalisation to intermediate values is good.** On the band it never saw, the ablation model is almost as accurate as the main model (0.92% vs 0.86%). Its uncertainty also stayed calibrated (95% coverage 96%). An "unseen k" of a new material is not a problem.
5. **Uncertainty calibration:**
   - ln G_c almost ideal.
   - ln k slightly conservative.
   - ln α clearly conservative: the 68% interval contains the true value 89% of the time. So the α uncertainty is wider than needed. It can be corrected with a single scale factor fitted on val; it is on the safe side, so not urgent.
6. **The trust checks alone are not sufficient.**
   - Thin samples and drift jumps are caught well.
   - Only 53% of the air-gap measurements and 4% of the strongly asymmetric contacts are flagged.
   - In unflagged faulty measurements the median k is 14% low.
   - Two checks of plan 9.8 are still missing: "is the predicted G_c within the training range" and "re-simulation residual" (M0+). The trust system must be completed before moving to real measurements.

## Addendum: trust checks v2 (without retraining the model)

After Phase 6, new checks were added on the outputs of the frozen M2 v1. Code: `src/tpslib/eval/trust.py`, settings: `configs/trust.yaml`. **All thresholds and scales are chosen on val only.**

**New checks:**
- **Training-range check (plan 9.8):** flagged if the predicted ln G_c, ln k or ln α lies outside the range seen on normal val measurements (both ends, e.g. below the 0.5% quantile). Expected for the air gap: the true G_c is lower than anything seen in training, and the model predicts it at or below the lower end of the range (not verified yet).
- **Ensemble spread for every target:** v1 only looked at ln k; now the member spread of ln α and ln G_c is checked as well.
- **Shared false-alarm budget:** the number of checks grew; so that false alarms do not grow with it, all checks are tuned with a single tail share. The total false alarm on val stays below 5% (v1: ~4.7%).
- **Variance scale:** the std of each target is multiplied by a factor found on val (50/68/90/95% intervals brought to nominal coverage). The main target is the too-wide ln α interval (68% → 89%).
- **M0+ (re-simulation):** designed only, not run: `docs/M0PLUS.md`. Estimated time per measurement ~50–75 s on one core. It will be the real answer to strongly asymmetric contact; v2 is not expected to catch it well enough.

**Re-run:** `python scripts/evaluate.py --trust-only ...` (Colab cell in `docs/COLAB.md`). Outputs in `reports/phase6_trust/` (`TRUST_REPORT.md`, `summary.json`, `trust_v2.json`, figures).

> **This run is the second use of the test sets.** The first use is the Phase 6 results above. The new thresholds and scales were not chosen by looking at the test set; every table in the report carries this label.

**Results (Colab run 2026-10-19).** The full evaluation was repeated in the same session with English figures; every Phase 6 number above was reproduced exactly (same models, same seeds).

| Case | n | v1 (Phase 6) | v2 |
|---|---|---|---|
| Air gap | 670 | 52.8% | **74.3%** |
| Thin sample | 674 | 97.9% | **99.3%** |
| Drift jump | 656 | 84.9% | **92.1%** |
| Contact asymmetry > 10 | 1,285 | 4.2% | 5.2% |
| False alarm (normal measurement, test) | 3,715 | 4.7% | 5.7% |

- v2 thresholds: shared tail share 0.0057 per check, total false alarm on val 5.0% (budget 5%). On test the false alarm is 5.7%, slightly above the val budget; this is the normal spread between two 5,000-sample sets.
- **The range check on G_c is what lifts the air-gap detection:** it alone flags 47% of the air-gap cases. The per-target ensemble spread (ln α) lifts the drift-jump detection. Plan targets of 90% are now met for thin sample and drift jump, not for air gap.
- Range seen on val (SI): k 0.0514–28.9 W/m·K, α 1.33e-8–2.19e-5 m²/s, G_c 1,090–13,900 W/m²·K.
- **What is left unflagged is harder:** the 229 faulty measurements that v2 does not flag have a median k error of −28% (v1: 429 unflagged, −14%). These are mostly air gaps with a G_c just inside the val range.
- **Asymmetric contact is still not caught (5%).** As expected, output-based checks cannot see it; M0+ (separate h_c fit) remains necessary before real measurements.

**Variance scale** (fitted on val; multiplies the predicted std):

| Target | Scale | test 68% raw → scaled | test 95% raw → scaled |
|---|---|---|---|
| ln k | 0.898 | 74.7% → 69.3% | 97.0% → 95.1% |
| ln α | 0.589 | 89.3% → 70.7% | 99.3% → 93.1% |
| ln G_c | 0.974 | 69.1% → 67.8% | 96.4% → 96.0% |

Figures: `phase6_trust/faulty_detection.png` (v1 vs v2) and `phase6_trust/calibration.png` (raw vs scaled coverage).

With the scale all three targets are close to nominal coverage on test; the ln α interval shrinks by 41%. The ONNX graph is unchanged; the scale is in `trust_v2.json` and is applied as `std × scale` on the user side.

## Findings carried over from Phase 5

- **Contact asymmetry decision.** By the user's decision, a contact difference larger than 10× between the two faces counts as a fault. The training domain was filtered accordingly (37,604 samples). In the realistic contact domain this lowered the high-k p95 error from ~13% to ~9% and the α median from 3.7% to 1.5% (Phase 5 diagnosis). Once the real contact distribution is learned from the first measurements, it will be re-evaluated with data v2.
- **Ensemble consistency.** The 5 members are very close to each other on val: k median 1.05–1.12%, p95 6.1–6.4%. Training is stable.

## Next steps (proposal)

1. **Complete the trust system** (before real measurements):
   - G_c and k range checks
   - M0+: nonlinear least-squares inversion with the simulator + re-simulation residual
   - calibration of the α uncertainty scale on val
2. **Device side** (plan WP1–WP4): sensor order, bring-up of the measurement chain (T1–T4).
3. **First real data** (sim-to-real, plan 9.7):
   - Idle and dummy-load recordings → real values in `device.yaml`.
   - First measurements on PMMA and glass → check whether the simulator reproduces the real curve without leaving structure in the residual, and the real contact distribution.
   - Then, if needed, data v2 and retraining; after that, fine-tuning.
