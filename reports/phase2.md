# Phase 2 report — Simulator and validation (2026-10-17)

> All results are **simulation**. This phase shows that the simulator is numerically correct and produces the expected physics. How well the simulator represents the real device will only be known from real measurements (plan section 9.7).

## Gate summary

| Gate | Target | Result | Status |
|---|---|---|---|
| 1. Analytic limit (no layers, perfect contact, constant power, 80 mm sample) | within 0.5% | at most **0.26%** for 0.1 ≤ τ ≤ 1; M0 k error ≤ 0.018%, α ≤ 0.056% | ✔ |
| 2. Convergence (mesh and time step halved) | < 0.1% change | **0.093–0.099%** for 0.1 ≤ τ ≤ 1; M0 k ≤ 0.019%, α ≤ 0.042% | ✔ (close to the limit) |
| 3. Energy conservation | within 0.5% | discrete balance **~10⁻¹⁶**; also **< 10⁻¹⁵** against an independent trapezoidal integral of P(t) | ✔ |
| 4a. Apparent k drops as h_c drops | monotonic | monotonic for 3 materials (−10% for steel at h_c=500) | ✔ |
| 4b. Late deviation for thin samples | visible | 8 mm: after τ≈0.59, +2.6–3.0% at τ=1; < 0.01% at 16 mm and above | ✔ |
| 4c. Layers cause an early-time deviation | visible | slope deviation exceeds 1% up to τ ≈ 0.13–0.78 (depends on h_c); a constant offset remains afterwards | ✔ |
| 5. Speed | < 1 s | **2.1–2.7 s** (one core) | ✘ |

**The speed gate did not pass.** This was left deliberately because accuracy has priority (rationale below). On one core, 60,000 simulations take about 39 h. In parallel runs on this laptop **memory bandwidth** sets the limit (Phase 3 pilot): throughput saturates at ~1.5–1.6 simulations/s with 6–12 processes, i.e. **~11–12 h** for 64,000 samples.

Tests: **78 passed** (13 of them simulator tests). To reproduce: `python scripts/validate_sim.py`. Outputs under `reports/phase2/`.

## Structure of the simulator

- **Geometry:** axisymmetric r–z plane, two samples.
- **Sensor layers:** four layers between the samples: PI base 25 µm | copper level 18 µm (copper in the traces, adhesive between them) | adhesive 15 µm | PI cover 12.5 µm.
  - Each layer is modelled as lumped resistance and heat capacity, with radial conduction within the layer.
  - Separate h_c on the bottom and top faces.
  - Film Ø20 mm; outside the film there is an air gap between the samples.
  - Convection h = 10 W/m²·K on the outer surfaces of the samples.
- **Source and readout:** 20 rings, Joule heat proportional to ring length. The read temperature is the length-weighted mean. The constant-current feedback P = I²R₀(1 + α_R·ΔT) is solved implicitly within the step.
- **Time:** TR-BDF2 (γ = 2−√2; L-stable, second order). The first level uses implicit Euler with Richardson extrapolation to damp the power step.
  - Both stages of TR-BDF2 use the same matrix: 2 back-substitutions per step (3 with Richardson).
  - dt doubles every 12 steps; one LU factorisation per level is enough.
  - Sparse LU (SuperLU, symmetric minimum-degree ordering).
- **Output:** ΔT(t) and P(t) on a 512-point log-time grid, plus an energy breakdown.

## Lessons from the mesh study (important)

1. **There is a ~20 µm boundary layer at the copper trace edge.** The copper leaks heat sideways into the neighbouring adhesive and PI; the layer width is √(k_layer·t/U) ≈ 22 µm.
   - A coarse mesh with 1 cell per trace cannot resolve this layer. The result is a constant ΔT offset: 1.5% for PMMA, **15%** for steel.
   - Fix: 1.5 µm cells at the edge with growth 1.15.
   - This offset barely affects k, but it sets the size of the thin-layer effect at high k. It matters for the h_c estimate and for the fit to real data.
2. **The far-field growth ratios set the late-time shape and therefore α.** 1.05 instead of 1.15 was chosen for the radial growth outside the sensor and for the z growth in depth. This reduced the M0 α error from 0.9% to 0.02%.
3. **Two-zone mesh.** Fine columns are used only within 0.3 mm of the sensor plane, coarse columns further away.
   - The two zones are coupled by a linear-interpolation, symmetric and conservative interface.
   - The piecewise-constant interface I tried first gave 1–5% error in k and was rejected.
   - This structure lowers the cost about 4× at the same accuracy.
4. **α is sensitive to the time step.** With Richardson-Euler at 8 steps per level the ΔT difference is only 0.01%, but M0 α shifts by 0.1%; 16 steps were needed. In Phase 3 the scheme moved to TR-BDF2 to raise parallel throughput. 12 steps keep α within 0.02% of a 64-step reference and halve the number of solves.
5. **Conclusion:** the < 1 s target and 0.1% convergence could not be met together. Accuracy was chosen. A coarser mesh (around 1 s) gave a 15% offset for steel and 0.2% α error.

## Physical findings (why M2 is needed)

According to `physics_hc.png`, `physics_thickness.png` and `physics_layers.png`, classic M0 makes systematic errors under realistic conditions:

| Case | M0 k error | M0 α error |
|---|---|---|
| PMMA, h_c = 5000 | −0.07% | −0.9% |
| Borosilicate, h_c = 5000 | −0.8% | +0.3% |
| Stainless steel 304, h_c = 5000 | **−1.7%** | **+3.0%** |
| Stainless steel 304, h_c = 500 (poor contact) | **−10%** | **+28%** |
| PMMA, 12 mm sample | +0.9% | **−12%** |
| PMMA, 8 mm sample | −6% | −55% |

Interpretation:
- **In high-k materials the layers and the contact break M0.** This is the opposite direction to the effect Zhang et al. report for low k: at high k the sensor's own thermal resistance becomes dominant. M2 has to learn this effect; that is why h_c and the layer parameters are in the prior.
- **α is much more sensitive than k.** When the sample gets slightly thinner, ΔT deviates by only 0.2% but α shifts by 12%. The probing-depth check in the trust system (plan 9.8) is therefore critical.
- **Limiting thickness.** At 16 mm and above the finite-size effect is < 0.01%. The 8–25 mm range in the prior also covers faulty-measurement examples.

## Figures and tables

- `phase2/gate1_analytic.png` — simulator, finite-trace-width analytic solution and D(τ) + const; relative difference.
- `phase2/gate2_convergence.png` — change when mesh + time and time only are halved.
- `phase2/physics_hc.png`, `physics_thickness.png`, `physics_layers.png` — physical behaviour.
- Tables: `gate1_analytic.csv`, `gate2_convergence.csv`, `gate3_energy.csv`, `physics_*.csv`, `speed.csv`, `summary.json`.

## Limitations and open points

- **Smoothness of the derivative.** The derivative dΔT/d ln t has small jumps of ≤ 1% of the slope at the time-level transitions and at the last points of the record (the fine kinks in `physics_layers.png`). The measurement noise in M2's dθ/d ln t channel will be far above this.

- **Early times (τ < 0.1).**
  - At the analytic limit the difference rises to 1–4% at the earliest moments (the very thin near field of the finite-band solution).
  - The halving change goes up to 0.2% at τ ≈ 0.03.
  - The gates were defined on the window 0.1 ≤ τ ≤ 1. M2 will see the whole curve; the 0.2% numerical uncertainty in this early region stays below the measurement noise, but I note it in writing.
- **Physics not modelled.** Lead heating (plan: R_lead/R₀ ≈ 0.4%), the PETG plate and temperature-dependent material properties are not in the model. They will be added if needed in the residual comparison of plan 9.7.
- **Speed.**
  - Acceptable for data generation: 60k simulations take one night.
  - Slow for M0+ (inversion with the simulator, hundreds of simulations per measurement). A coarser "fast mesh" plus this mesh in the last step could be used there; to be handled after Phase 6.
