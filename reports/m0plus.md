# M0+ report — Re-simulation inversion and check (2026-10-19)

> **Development results on the pilot split only** (1,000 measurements, the val stand-in used during development, plus the 48-measurement `dev_faulty` set). No test split was used. All results are simulation.

## What was built

- `src/tpslib/eval/m0plus.py`
  - **Full inversion `fit_m0plus`.** Fits ln k, ln α, ln h_c,bottom and ln h_c,top with the r–z simulator (trust-region least squares, forward-difference Jacobian, three contact-split starts). At every evaluation a linear part is solved in closed form:
    - an amplitude scale c (absorbs the α_R, power and radius calibration errors; Gaussian prior σ = 1.2% around 1),
    - the heating-start offset t_c (first order),
    - an offset,
    - a residual linear drift.

    Cost ~100 simulations per measurement.
  - **Screening `m0plus_light`.** Re-simulates the measurement at M2's estimate for five contact splits (ln h_b/h_t = −3 … 3 at M2's G_c), 5 simulations. Score: the smallest χ²_ν.
- `scripts/run_m0plus.py`: M2 (ONNX) + M0 + M0+ screening on every measurement of a split, trust v3 thresholds from val, resumable.
- `configs/m0plus.yaml`, trust v3 section in `configs/trust.yaml` (v2 + the M0+ check, same shared false-alarm budget).
- `tests/test_m0plus.py`: contact split, noise estimate, a zero residual at the true parameters, the screening seeing a wrong contact split, and full recovery on a clean record (slow test, `TPS_SLOW_TESTS=1`).
- The film of each synthetic measurement is given to M0+ as known. This is the "sensor calibrated once" assumption of the design.

## Key finding 1 — the measurement chain must be in the residual model

Without the calibration scale and t_c, χ²_ν at the **true** parameters was 100–4,000 at high k. Calibration errors and t_c are not curve shape, but the fit then distorts k to absorb them (k errors up to −48%). With the linear nuisance part, χ²_ν at the true parameters is 0.2–2.6 for all 12 test cases, i.e. at the noise level.

## Key finding 2 — full inversion on 12 pilot measurements (start: true values with an M2-like error)

| k [W/m·K] | true asymmetry | M0+ k error | M0+ α error | fitted asymmetry | χ²_ν |
|---|---|---|---|---|---|
| 0.06 | 25 | +0.7% | +3.2% | 1.0 | 1.27 |
| 0.06 | 52 | −0.4% | −1.9% | 1.0 | 2.25 |
| 0.08 | 22 | +0.6% | −0.6% | 1.1 | 0.87 |
| 0.46 | 1.4 | −1.4% | 0.0% | 1.1 | 0.24 |
| 0.52 | 2.4 | −1.5% | +0.2% | 2.7 | 0.42 |
| 5.3 | 1.3 | −0.5% | +1.7% | 1.0 | 0.16 |
| 8.2 | 26 | −0.1% | +0.1% | 22.5 | 0.28 |
| 9.9 | 1.6 | **+61%** | −4.3% | **13.5** | 0.37 |
| 10.6 | 31 | +4.2% | +1.6% | 40.7 | 0.19 |
| 11.4 | 1.5 | −10.4% | −1.1% | 2.0 | 0.90 |
| 23.8 | 18 | +3.6% | +2.1% | 21.0 | 0.40 |
| 27.4 | 34 | −5.8% | −2.4% | 25.2 | 0.22 |

- At **low k** the contact barely matters: the fitted asymmetry is meaningless, but k is right anyway (as M2's is).
- At **high k** the inversion often finds the true asymmetry and a better k than M2 (−14% median for these cases). It also finds a false asymmetric solution with a 61% k error for one symmetric case, at the same χ²_ν. **Several parameter sets explain the curve equally well.**

## Key finding 3 — screening on all 1,000 pilot measurements

χ²_ν of the best contact split at M2's estimate (median / p95):

| | k 0.05–0.5 | k 0.5–5 | k 5–30 |
|---|---|---|---|
| normal (asymmetry ≤ 10) | 0.9 / 3.6 | 1.2 / 28 | 3.2 / 40 |
| asymmetry > 10 | 1.3 / 13 | 6.9 / 42 | 5.3 / 20 |
| faulty (`dev_faulty`: air gap, thin sample, drift jump) | 197 / 4,990 | 66 / 5,500 | 81 / 580,000 |

- **Real faults usually stand out:** median χ²_ν 70 (air gap, n = 14), 304 (thin sample, n = 15) and 66 (drift jump, n = 19), against ~1–3 for normal measurements. Above the v3 threshold (χ²_ν ≈ 100, set by the tail of normal measurements): air gap 43%, thin sample 80%, drift jump 32%; 50% of `dev_faulty` overall. Trust v3 (v2 + M0+) flags thin samples 100% (v2: 93%), with the same false alarm (4.9% on the pilot). The `dev_faulty` set is small, so these rates are rough.
- **Asymmetric contact is not separable.** With the contact split free, the asymmetric measurements fit M2's (biased) answer as well as normal ones. At the 99% threshold the screening flags 0–2% of them, and 5% in the 5–30 band with the symmetric split only. M2's k estimate plus a symmetric contact explains the curve within the noise.
- The size of the fitted split cannot be used either: |ln split| ≥ 3 flags 20–71% of asymmetric cases, but also 14% of normal ones.

## Conclusion

1. **Asymmetric contact at high k cannot be detected from a single measurement**, by M2, by the output checks of trust v2, or by re-simulation. The mean-temperature curve has an equally good explanation with a different k and a symmetric contact. This is a property of the measurement, not of the analysis.
2. **It has to be prevented or measured by the procedure.** Two options, to be verified with the simulator before the hardware is fixed:
   - Measure twice with the sensor **flipped**. The film is not symmetric (25 µm PI vs 12.5 µm PI + 15 µm adhesive), so an asymmetric contact gives two different curves; symmetric contact gives two nearly identical ones.
   - A fixture that guarantees equal clamping and a thin, repeatable interface (paste or foil) on both faces.
3. **M0+ is kept as a tool**, not as a default trust check:
   - `fit_m0plus` is a physics-based second opinion for flagged high-k measurements and for the sim-to-real check on real data (plan 9.7: χ²_ν ≈ 1 on PMMA and glass means the simulator explains the real curve).
   - Trust v3 is implemented and works, but on the pilot it adds little over v2 (v2 already flags most faults), and a full evaluation costs ~5 h of Colab CPU (5 simulations × 12,000 measurements). **Trust v2 stays the shipped trust system for M2 v1.** v3 can be evaluated later with `scripts/run_m0plus.py` if real data show faults that v2 misses.

## Files

- `reports/m0plus/m0plus_chi2.png`: χ²_ν distributions on the pilot (normal, asymmetric, faulty).
- Reproduce: `python scripts/run_m0plus.py --onnx models/m2_v1/m2_ensemble.onnx --splits val test_faulty --split-files val=pilot.h5 test_faulty=dev_faulty.h5` (~50 min on 12 local processes).
