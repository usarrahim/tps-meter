# Phase 1 report — Analytic solution and M0 (2026-10-17)

> All results in this phase are based on **analytic, noise-free synthetic data**. They show that the code is correct (test T5 of the plan); they do not show the real accuracy of the device.

## Summary

| Gate | Target | Result | Status |
|---|---|---|---|
| D(τ), the 6 values in PROJECT_RULES.md | within 0.05% | largest deviation **0.0007%** | ✔ |
| Table + interpolation vs direct computation | — | largest absolute difference **2.6·10⁻¹³** | ✔ |
| M0, 20 k values (0.05–30 W/m·K), constant power | k and α < 0.5% | k ≤ 0.0002%, α ≤ 0.001% | ✔ |
| M0, same 20 k values, constant current (real device condition) | k and α < 0.5% | k ≤ **0.003%**, α ≤ **0.019%** | ✔ |
| pytest | clean | **65 passed** | ✔ |

A single fit takes at most 0.07 s (including the 74,000-sample PMMA record).
To reproduce: `python scripts/validate_m0.py`. Outputs: `reports/phase1/`.

## What was done

- `src/tpslib/dtau.py`
  - Gustafsson D(τ), parameterised by m; `i0e` is used to avoid overflow.
  - Direct (adaptive quadrature) computation and a fast table: Simpson on a ln σ grid + cubic spline. The table is built once in 0.16 s.
  - Ideal temperature rise for two cases: constant power, and **constant current** (P = I²R₀(1+α_R·ΔT), solved by Duhamel superposition).
- `src/tpslib/fit_tps.py` (M0)
  - Model: ΔT = A·D̃(t) + B·P(t)/P₀.
  - D̃ is D weighted by the power history. At constant power D̃ = D(τ) and the model reduces to the classic ΔT = A·D(τ) + B.
  - Window 0.3 < τ² < 1. α search: first a coarse log scan, then a 1D minimisation with the window fixed and a window update, until the window no longer changes.
  - k = P₀ / (π^1.5·a·A).
  - The result also returns the residual RMS, the window, τ_max and a "window truncated" warning.
- `src/tpslib/synth_analytic.py`: test curves.
  - Simplified auto-setup with the true k and α: power targeting 2.5 K at τ=1 (clipped to 8 mW–1 W), duration up to τ=1.
  - Sample rate: the highest rate allowed by the budget of 12,000 samples per record.
- `configs/analysis.yaml`: D(τ) and M0 settings. The sample budget was added to `device.yaml`.
- Tests: `tests/test_dtau.py`, `tests/test_fit_tps.py`.

## Key finding 1 — Lower limit of D(τ)

- As σ→0 the integrand of D(τ) tends to a constant value (1/(2√π(m+1)) = 0.0134) in ln σ. The integral therefore **diverges logarithmically**: the self-temperature of an infinitely thin ring is infinite (`dtau.png`, right panel).
- The reference values in PROJECT_RULES.md are reproduced exactly with the lower limit **σ_min = 10⁻⁶**. This value is written in the config.
- As a result, the absolute value of D depends on σ_min. The dependence enters only as a constant term. In M0 this term goes into B; k and α are not affected at all (`test_cutoff_independent`).
- Effect on Phase 2: the simulator will not be compared with the analytic solution through the absolute ΔT, but in one of two ways:
  - a reference regularised by the finite trace width b,
  - or the recovery of k and α.
- Details: `docs/DECISIONS.md`.

## Key finding 2 — Power change at constant current

The device measures at constant current; the power rises by about 1% as it heats. If this rise is ignored and the mean power is used (`m0_constant_current_effect.png`):
- k comes out **0.66–0.73%**,
- α **1.9–2.2%**

systematically low. This error alone uses up most of the simulation target (k median < 1%). M0 therefore uses the measured P(t); the error drops to 0.003%.

## Figures

- `phase1/dtau.png`: D(τ) and the reference points; behaviour as σ→0.
- `phase1/m0_gate_errors.png`: k and α error of M0 at 20 k values, in both power modes.
- `phase1/m0_constant_current_effect.png`: the error that appears if P(t) is not taken into account.
- `phase1/m0_example_fit.png`: PMMA-like example; ΔT–D̃ line and the residual within the window (±20 µK, from the curve interpolation).
- Tables: `phase1/dtau_reference.csv`, `phase1/m0_gate.csv`, `phase1/summary.json`.

## Limitations and next steps

- M0 does not fit t_c (start-time offset) yet. Once noise, drift and t_c are added in Phase 4, M0's < 2% gate will be tested there. A t_c parameter will be added if needed.
- In a real measurement B carries the temperature difference across the Kapton layers. The early-time deviation caused by the layer heat capacity will be seen with the Phase 2 simulator; whether the window (0.3 < τ²) excludes it will be measured there.
