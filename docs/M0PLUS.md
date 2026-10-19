# M0+ — re-simulation inversion and residual check (design, not implemented yet)

The last trust check of plan 9.8. Goal: re-run the simulator with M2's estimate and see whether it explains the measured curve down to the noise level. If it does not, something lies outside the measurement or model domain (air gap, strongly asymmetric contact, unmodelled physics).

The checks in trust v2 (range, ensemble spread, M0 difference) stay weak especially for **strongly asymmetric contact**: M2 gives a confident, in-range answer for these curves too. M0+ is needed to close this gap.

## Unknowns

| Parameter | Type | Start | How it is solved |
|---|---|---|---|
| ln k, ln α | nonlinear | M2 estimate | Levenberg–Marquardt (LM) |
| ln h_c,bottom, ln h_c,top | nonlinear | symmetric split of M2's G_c | LM (two separate parameters) |
| B (constant offset + remaining linear drift slope) | linear | — | closed-form least squares at every LM step (variable projection) |
| t_c (heating start offset) | nonlinear but cheap | 0 | shift of the simulated curve along the time axis (spline), no new simulation needed |

- Keeping the two h_c separate matters: the fitted h_c,max/h_c,min ratio becomes the "asymmetry > 10" check directly.
- Film layer thicknesses and properties are held at the prior mean (calibrated once on the real device).

## Algorithm (per measurement)

1. The preprocessed record (`preprocess` output: t, ΔT, P) comes through the same path as for M2.
2. Compute the residual in 256 log-time bins (same grid as the features). The noise σ_i of each bin comes from the pre-measurement noise floor and the number of samples per bin.
3. LM: 4 nonlinear parameters, forward-difference Jacobian (step 1e-3, in ln space). t_c and B are solved inside every evaluation.
4. Stop: relative χ² improvement < 1e-3 or 8 iterations.
5. Outputs:
   - χ²_ν = mean((r_i/σ_i)²) → **residual check**, threshold from the 99th percentile on val.
   - Fitted h_c ratio → **asymmetry check** (fault if > 10).
   - Fitted k, α → M0+ estimate. Its difference to M2 is also a check.

## Time estimate

- Single simulation (Phase 2 speed measurement, this laptop, one process): 2.1–2.7 s (`reports/phase2/speed.csv`).
- LM iteration: 1 centre + 4 Jacobian simulations = 5 simulations. With M2's good start, 3–5 iterations plus 1–2 rejected steps → **~20–30 simulations**.
- **Per measurement: ~50–75 s on one core.** With the Jacobian spread over 5 cores, ~12–18 s wall time.
- Speed-up options:
  - Coarse mesh (half radial/axial resolution) in the first iterations, fine mesh in the last: ~2×.
  - "M0+ light": a single simulation with M2's estimate (~2.5 s) and χ²_ν. Expected to catch the air gap; it does not catch asymmetry (the h_c split is unknown).
- Suitable for device use: the measurement itself takes 5–800 s; 15–75 s extra on the PC side is acceptable.

## Validation plan (Colab)

- Threshold: 500 normal measurements from val (~500 × 60 s ≈ 8 CPU-hours → ~20 min on 24 cores).
- Effect: 200 of each fault type from test_faulty, 300 asymmetry > 10 samples from test.
- Because this would be the third use of the test sets, evaluate first within val using its asymmetry > 10 samples; test only in the final report.
- First use on real data: if χ²_ν ≈ 1 for PMMA and glass, the simulator explains the real curve without leaving structure (the same test as the sim-to-real check in plan 9.7).
