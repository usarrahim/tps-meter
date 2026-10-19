# Design decisions

Each entry: date, decision, rationale. Open questions become decisions once the user answers them.

## 2026-10-17 — Phase 0

- **Python 3.11 (local), code compatible with ≥3.11.** The computer only had 3.14; as PROJECT_RULES.md requires, 3.11.9 was installed with the Python install manager. Colab currently uses 3.12, so `requires-python = ">=3.11"`; no 3.12-specific syntax is used.
- **Local PyTorch with the CUDA 12.8 wheel** (RTX 3050 Ti 4 GB, driver CUDA 13.1). `pyproject.toml` lists torch without a version; locally the CUDA wheel is installed first, Colab uses its own torch.
- **Project root `tps-master/`** (the existing folder instead of the `tps-meter/` name in the brief).
- **Data and output roots can be changed by environment variables** (`TPS_DATA_ROOT`, `TPS_RUNS_ROOT`, `src/tpslib/runtime.py`). On Colab the data goes to the local disk and the outputs to Drive.
- **Device selection** `tpslib.runtime.get_device()`: CUDA if available, otherwise CPU; can be forced with `TPS_DEVICE=cpu|cuda`.
- **Placeholder values:** the `device.yaml` noise model and current ranges, the nominal ρc_p of the adhesive/PI in `sensor.yaml`. They will be replaced when real values arrive; they are marked `PLACEHOLDER` in the files.

## 2026-10-17 — Decisions on the Phase 0 open questions

User answer: "Design the plan yourself, no need to scale anything down, do it the best way, accuracy first, upload to Drive as a zip."
The project plan PDF (Design and Implementation Plan, Chapters 2, 9, 10) was taken as the reference.

1. **h_c output → total interface conductance `G_c`.** The sensor only reads the mean temperature, so the bottom and top h_c are each weakly identifiable; what is identifiable is the total thermal conductance from the copper into the two samples:
   `G_c = 1/(R_bottom_layer + 1/h_bottom) + 1/(R_top_layer + 1/h_top)` [W/m²·K]. The model outputs `log G_c` (weight 0.2). Bottom/top h_c are stored separately in the dataset and can become targets later if needed.
2. **M2 architecture as described** (4 blocks, 32–64–128–128, kernel 5, 2 convolutions per block, ≈330k parameters). "~200k" was only an estimate; nothing is scaled down. The compact device model (~60k) stays in the plan but does not affect the desktop result.
3. **test_faulty:** the file holds the clean curve + a fault label; the drift jump is added at evaluation time with a fixed seed through the same `acquisition` path.
4. **Energy balance:** `∫P dt = Σ (ρc_p·V·ΔT)_all cells (samples + layers + copper) + ∫ outer-surface convection loss dt`, relative difference < 0.5%. The share stored in the samples is reported separately.
5. **M0 window 0.3 < τ² < 1** (PDF Chapter 2: literature rule, Zhang et al.). Adjustable in the config; for thin samples (glass 10 mm) it can be reduced to τ ≤ 0.75.
6. **Accuracy > speed.** The convergence gate comes first; if the < 1 s target is not met, parallel generation compensates and the time is reported.
7. **Colab:** code as a zip package (`scripts/pack_for_colab.py`). Local time is measured first; if local is reasonable, work stays local.

## 2026-10-17 — Lower integration limit of D(τ) (Phase 1)

- **Finding:** the integrand of D(τ) behaves like `1/(2√π(m+1)σ)` as σ→0; the integral diverges logarithmically (the self-temperature of an infinitely thin ring is infinite). The PROJECT_RULES.md test values (D(0.1)=0.20553 …) are reproduced exactly with the lower limit **σ_min = 1e-6** (0.2055295).
- **Decision:** the `dtau` lower limit is `sigma_min: 1.0e-6` in the config. This choice only adds a constant term to D: `c = ln(1/σ_min)/(2√π(m+1))`. At constant power this term goes into B; at constant current it varies as P(t)·c, so the M0 model is `ΔT = A·D̃(t) + B·P(t)/P₀` (D̃: D weighted by the power history through Duhamel). The cut-off constant therefore never affects k or α.
- **Consequence:** the absolute "ideal" ΔT (`P/(π^1.5·a·k)·D`) depends on σ_min and is not physical. In Phase 2 the simulator–analytic comparison is made against a reference regularised by the finite trace width b, or through an (A, B) fit, not against the absolute D value.

## 2026-10-17 — Scope extension (output aimed at researchers)

The user asked for "output that truly serves researchers". Following PDF sections 9.5 and 9.8, these parts that were not in the brief will be added (proposed as a separate phase after Phase 6):
M0+ (full-physics inversion with the simulator), a trust system (probing depth, ΔT range, re-simulation residual, ensemble and method agreement), result format `k = … ± … W/m·K (95%), α = … ± … mm²/s, status + reason` (JSON/TXT/PDF).
- The PDF's "ambient temperature" scalar input cannot be simulated meaningfully in synthetic data, so it is left out of M2 for now; it will be added during fine-tuning with real data.

## 2026-10-17 — Phase 2: simulator design

- **Thin-film stack (bottom to top):** PI base 25 µm | copper level 18 µm (copper in the traces, coverlay adhesive between them) | adhesive 15 µm | PI cover 12.5 µm. Each layer has one node per radial column (lumped resistance + capacity) and radial conduction within itself. Film radius 10 mm (plan: Ø20 mm active area); outside the film there is an air gap of film thickness between the two samples.
- **Heat source and readout:** the copper cross-section is the same in every ring, so Joule heat is spread uniformly over the trace volume (ring power ∝ ring length). The readout uses the same weights, i.e. the length-weighted mean.
- **Constant current:** P = P₀(1 + α_R·T̄) is solved implicitly within the time step; the rank-1 term is handled with Sherman–Morrison.
- **Time:** first, implicit Euler with Richardson extrapolation was used (16 steps/level; 8 steps shifted M0 α by ~0.1%). The Phase 3 pilot showed that parallel generation is limited by memory bandwidth. So the scheme was switched to **TR-BDF2 (γ = 2−√2)**: both stages use the same matrix, 2 solves per step are enough, and 12 steps/level give the same accuracy. The first level uses implicit Euler + Richardson to damp the stiff modes of the power step. dt doubles at every level, so one LU per level is enough. First step 0.1 ms; for slow records `dt0 = max(0.1 ms, 0.05·t_first_output)`.
- **Mesh (result of the mesh study, `reports/phase2.md`):**
  - The lateral leakage at the copper trace edges has a ~20 µm boundary layer. Cells of 1.5 µm are used at the trace edge; growth 1.15 in the gap and 1.4 inside the trace.
  - Two zones: fine (trace-aligned) columns within 0.3 mm of the sensor plane, coarse columns further away. The two zones are coupled by a linear-interpolation, symmetric and conservative interface (PᵀGP).
  - A piecewise-constant interface was rejected because it gave 1–5% k error.
  - Far-field z growth 1.05; radial growth outside the sensor 1.05. Late-time shape and α are sensitive to these two ratios.
- **SuperLU ordering `MMD_AT_PLUS_A` + `SymmetricMode`:** for the SPD matrix about 40% less fill-in than COLAMD and 2× faster solves.
- **Analytic reference:** D(τ) is defined for infinitely thin rings and its absolute value depends on σ_min (Phase 1). The analytic-limit gate is therefore run against the finite-trace-width band solution, which comes from the exact expression via a Hankel transform (`sim/analytic.py`). It is tested that it tends to D(τ) + const as b → 0.

## 2026-10-17 — Phase 3: prior and auto-setup

- **Auto-setup rule:**
  - Power: 2.5 K target at τ = 1, using the probe k and the ideal formula; clipped to 8 mW–1 W.
  - Time: `min(a²/α_probe, probing depth = 0.9·min(thickness, radius − a))`. The probing-depth rule is the same as the trust check in plan 9.8.
  - Sample rate: the highest rate allowed by the 12,000-sample-per-record budget.
  - Current range: L/H from plan 5.3.
  - The calibrated radius is used.
  - Note: the ideal formula does not see the film/contact offset, so ΔT_max reaches 3–10 K at high k. If the firmware chooses the power from the measured ΔT, the rule must be updated (open question, `reports/phase3.md`).
- **Faulty measurements:** one fault per sample.
  - Air gap: h_c 20–300.
  - Thin sample: true thickness 2–6 mm, the entered value is wrong.
  - Drift jump: added in the measurement chain.
- **Generation layout:** atomic HDF5 shards of 250 samples, resume, merge into the split file, manifest. Because there was no git commit, a code hash (sha256) is recorded as well.
- **Time scheme:** TR-BDF2 (2 solves) instead of Richardson-Euler (3 solves per step), because of the memory-bandwidth limit in parallel generation.
- **YAML:** PyYAML reading the `1e6` form as text was fixed with a custom loader.

## 2026-10-18 — Phase 4: measurement chain and features

- **Raw record format:** measurement effects are not added directly to the clean curve but in the device's raw record format (ADC1 ratio, ADC2 current, pre-measurement R₀ and slope, calibration). Real and synthetic data therefore go through exactly the same `preprocess` path.
- **Noise model:** built on the ADC input noise (µV); its temperature equivalent scales with the current (1/I) (same as the table in plan 5.2). It is allowed that the real board may be up to 2× noisier than the datasheet (U(1, 2)).
- **θ scaling:** θ/θ_ref channel + ln θ_ref scalar. θ varies 600× with k; this scaling improves conditioning without losing information. ΔT/ΔT_max was rejected (it erases the k information); the Fourier number cannot be an input (α is unknown).
- **Extra channel:** ln(1 + samples per bin) was added (beyond the brief). It tells the model the noise level per bin.
- **Effect seeds:** in train the effects are new every epoch (seed = base, epoch, index); fixed in val and test.
- **Normalisation:** from train only; computing it on another split is blocked in the code.
- **Open point:** t_c ±5 ms is a placeholder; it changes the early times noticeably at high k. The relation between the DAC step and the DRDY timestamp must be measured in the firmware.

## 2026-10-18 — Phase 5: training and data domain

- **Early stopping only after epoch 30 (`min_epochs: 30`).** The brief's "patience 8" rule stopped the 50k run at epoch 28, while the learning rate was still high and the val metric noisy. The main improvement period after the cosine decay was missed. Patience 8 is kept, but it only acts in the second half of training.
- **Contact asymmetry (open decision, asked the user):** the prior draws bottom and top h_c independently; 30–100× asymmetry is common in the data. At high k these samples create an indistinguishable ambiguity (k underestimated by 20–30%). A model trained on asymmetry ≤ 10 reduced the high-k p95 error from 13% to 8.6% and the overall α median error from 3.7% to 1.5% in the realistic contact range. The comparison becomes fair once the model trained on all data is finished.
- **Fix:** the patience counter now only counts after `min_epochs`. The first version counted from the start, so the resumed 50k run stopped at epoch 30 again.
- **Decision (user, 2026-10-18): contact asymmetry > 10 counts as a fault.** The ensemble (`v1sym`) and the ablation are trained with `--max-asym 10`. The existing data are filtered (train 37.6k); v2 data will be generated once the realistic contact distribution is learned from real measurements. The full comparison with all data was skipped: model (b) is good enough in its own domain and the decision rests on a physical argument.

## 2026-10-18 — Phase 6: evaluation design

- **Test domain:** the main results are reported on the realistic contact domain of the test set (asymmetry ≤ 10). Samples with asymmetry > 10 are shown on a separate row ("fault domain") and are expected to be flagged by the trust checks.
- **Trust checks:** ensemble spread, total uncertainty, M0–M2 difference, M0 residual and ΔT range. Thresholds come from the 99th percentiles on **val**; the test sets are never used to choose thresholds. The re-simulation check (M0+) is not in this version.
- **ONNX:** a single graph with the ensemble's normalisations built in; input raw features, output mean and std in ln units. The comparison with PyTorch runs automatically.
- **Compact model:** (16, 32, 48, 48) channels, kernel 5, head 64 → 61,910 parameters; 2 members as in plan 9.11.

## 2026-10-19 — Following the Phase 6 results

- **Base model M2 v1 frozen:** `v1sym` 5 members + compact `v1c` 2 members. The test sets have been used; no further model selection will be made for this version.
- **Trust system v1 is not sufficient** (air gap 53%, asymmetry 4%). To be added before real measurements:
  - G_c and k range checks
  - M0+ re-simulation residual
  - calibration of the α uncertainty scale on val

  These improvements do not require retraining the model; new thresholds will be chosen on val, and the test sets will be reused only once, in the final report.

## 2026-10-19 — Trust checks v2 (M2 v1 frozen, no retraining)

- **Range check:** the "training range" is not the range of the training labels but the range of ln k, ln α, ln G_c that the model *predicts* on normal **val** measurements. The model saturates at the boundary, so the label range would miss real faults; the prediction range is consistent with the model's own behaviour.
- **Shared false-alarm budget (5%):** in v1 each check was tuned separately at its 99th percentile; going from 5 to 10 checks would double the false alarms. In v2 a single tail share is found on val by bisection; the total false alarm stays at the v1 level. The budget is in `configs/trust.yaml`.
- **Variance scale:** one factor per target, the geometric mean of the exact factors at the 50/68/90/95% levels. Tuning to a single level could distort the tail. The ONNX graph was not changed (frozen model); the factor is in `trust_v2.json`.
- **Test use:** the `--trust-only` run is the second use of the test sets; this is stated explicitly in the report and in `summary.json` (`test_use`). No model or threshold choice will be changed based on the test result.
- **M0+:** design in `docs/M0PLUS.md`; the two h_c are fitted separately so that the asymmetry is measured directly. Implementation is a separate task.

## 2026-10-19 — Repository language

- **Everything in the repository is English** (user request): code comments, docstrings, plot labels, reports, model card, docs, notebook and PROJECT_RULES.md. The earlier PROJECT_RULES.md rule ("reports in Turkish") was changed accordingly. The chat with the user stays in Turkish. The main report was renamed `reports/RAPOR_v1.md` → `reports/REPORT_v1.md`.
- Figures of Phases 1–4 were regenerated locally with English labels (same seeds, same numbers). Phase 6 figures need the Colab data; they are regenerated with the next Colab evaluation run.

## 2026-10-19 — Trust v2 results

- **Trust v2 adopted** for M2 v1 with the val-fitted thresholds and variance scale in `trust_v2.json`. The test false alarm (5.7%) is slightly above the 5% val budget; accepted as sampling spread, the budget is not re-tuned on test.
- **M0+ stays required before real measurements:** output-based checks catch only 5% of asymmetric contacts, and the air gaps they miss carry a −28% median k error.

## 2026-10-19 — M0+ implemented; trust v2 stays the shipped trust system

- **Residual model:** the re-simulation needs a closed-form linear part (amplitude scale with a 1.2 % prior for the calibration errors, heating-start offset t_c, offset, residual drift). Without it χ²_ν at the true parameters is 100–4,000 at high k and the fit distorts k.
- **Finding:** at high k an asymmetric contact and a symmetric contact with a different k explain the curve equally well (pilot: the screening flags 0–5 % of asymmetric cases; the full inversion found one false asymmetric solution with +61 % k at the same χ²_ν). Asymmetric contact must be handled by the procedure (flipped second measurement or a fixture with equal clamping and a repeatable interface), to be checked with the simulator before the fixture is fixed.
- **Decision:** M0+ is kept as a tool (second opinion for flagged high-k measurements; sim-to-real check on real data). Trust v3 (v2 + M0+ check) is implemented but not adopted for M2 v1: on the pilot it adds little over v2 and its full evaluation costs ~5 h of Colab CPU. The test splits were not used for M0+.
- **Base model closed:** M2 v1 + trust v2 + M0+ tool are tagged `v1.0-base`.
