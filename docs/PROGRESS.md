# Progress

| Phase | Status | Gate | Report |
|---|---|---|---|
| 0 — Skeleton and environment | Approved | `pytest` clean ✔ | [reports/phase0.md](../reports/phase0.md) |
| 1 — Analytic solution and M0 | Approved | D(τ) 0.0007%; M0 k ≤ 0.003%, α ≤ 0.019% (20 k values, incl. constant current) ✔ | [reports/phase1.md](../reports/phase1.md) |
| 2 — Simulator and validation | Approved (scheme moved to TR-BDF2 in Phase 3, gates passed again) | Analytic 0.26% ✔, convergence 0.099% ✔, energy ✔, physics ✔, speed 2.1–2.7 s ✘ (deliberate) | [reports/phase2.md](../reports/phase2.md) |
| 3 — Prior and data generation | Done (Colab) | 64,000/64,000, 0 failures ✔ | [reports/phase3.md](../reports/phase3.md) |
| 4 — Measurement effects and features | Approved | M0 + measurement effects k ≤ 0.70% ✔; normalisation ✔ | [reports/phase4.md](../reports/phase4.md) |
| 5 — M2 training | Done (Colab): 5-member `v1sym` + ablation + compact | Val k median 1.05–1.12%, p95 6.1–6.4% ✔ | [reports/phase5.md](../reports/phase5.md) |
| 6 — Evaluation and export | Approved (base model M2 v1) | k<5: target ✔; 5–30: ✘ (p95 10%); ONNX ✔; fault detection partial | [reports/phase6.md](../reports/phase6.md) |

## Phase 0 — 2026-10-17

- Folder layout, `pyproject.toml`, `.gitignore`, 6 config files and module stubs set up; git repository initialised (no commit yet).
- `.venv`: Python 3.11.9 + PyTorch with CUDA; GPU: RTX 3050 Ti Laptop (4 GB).
- `tpslib.runtime.get_device()`: picks CUDA if available, otherwise CPU.
- `tpslib.config`: YAML loader + `SensorConfig` (with SI range checks).
- Colab: `notebooks/colab_run.ipynb`, `scripts/pack_for_colab.py`, `docs/COLAB.md`.
- Open questions asked to the user; decisions in `docs/DECISIONS.md`.

## Phase 1 — 2026-10-17

- `dtau.py`: direct D(τ) computation + table; ideal ΔT for constant power and constant current.
- `fit_tps.py`: M0, ΔT = A·D̃ + B·P(t)/P₀, window 0.3 < τ² < 1, α scan.
- Findings: lower limit of D(τ) σ_min = 10⁻⁶ (does not affect k); ignoring constant current gives 0.7% k and 2% α error.
- Tests: 65 passed. Report and figures: `reports/phase1.md`, `reports/phase1/`.

## Phase 2 — 2026-10-17

- Simulator: `sim/geometry.py` (trace-aligned, edge-refined, two-zone mesh), `sim/solver.py` (finite volume, implicit Euler with Richardson, constant current via Sherman–Morrison), `sim/sensor_model.py` (`simulate()`), `sim/analytic.py` (finite-trace-width analytic reference).
- `scripts/validate_sim.py`: five gates + figures. All gates except speed passed; speed 2.8–3.5 s/sim (accuracy chosen over speed).
- Mesh-study findings: ~20 µm boundary layer at the copper trace edge; α is sensitive to the far-field growth ratios and to the time step.
- Tests: 78 passed. Figure review with the user pending (mandatory stop in the brief).

## Phase 3 (pilot) — 2026-10-17

- `data/prior.py` (prior + auto-setup + faults + G_c), `data/generate.py` and `scripts/generate_dataset.py` (parallel, resumable, HDF5, manifest), `scripts/pilot_report.py`.
- Pilot: 1,000 samples, 0 failures, 1.44 samples/s. Full-run estimate ~12 h (bottleneck: memory bandwidth).
- Simulator time scheme moved to TR-BDF2 (+50% parallel throughput); Phase 2 gates passed again.
- Fixed and tested the bug where YAML values such as `0.8e6` were read as text.
- Tests: 90 passed. Waiting for user approval for the full run.

## Phase 3 (full generation) — 2026-10-17/18

- The user chose to run all generation on Colab. Drive backup/restore (`--mirror`) and a code-hash guard were added to the generator; `scripts/bench_generation.py` and the Colab notebook were written.
- v5e-1 TPU runtime (24 CPUs): 3.37 samples/s → ~5.3 h. Generation ran in the user's Colab session.

## Phase 4 — 2026-10-18

- `measurement/acquisition.py` (clean curve → raw device record, all measurement effects), `measurement/preprocess.py` (shared by real and synthetic), `data/features.py` (256 log bins, 3 channels, 7 scalars), `data/dataset.py` (new effects every epoch, normalisation from train only).
- Gate: M0 ideal curve + measurement effects k ≤ 0.70% (1.59% with calibration). For long, low-k measurements α is dominated by the ambient drift (5–10%).
- Tests: 125 passed. `device.yaml` changed: no new zip should be uploaded before the Colab generation finishes.

## Phase 5 (code) — 2026-10-18

- `models/resnet1d.py` (375k parameters), `models/losses.py` (β-NLL), `train/trainer.py` (AMP, cosine + warm-up, early stopping, resumable runs, epoch-encoding sampler), `scripts/train_m2.py` (seeds, ablation, data size), Colab cells 7–12.
- Pilot trial (900/100, 40 epochs, batch 32): k median 1.8%, α median 9%, cov95 96%. Code verified.
- Tests: 131 passed. Full training after the Colab generation finishes.

## Phase 5 (training) and Phase 6 (code) — 2026-10-18

- 10k model and diagnosis: the tail comes from high k and extreme contact asymmetry. User decision: asymmetry > 10 is a fault; ensemble `v1sym` (5 seeds, asymmetry ≤ 10) trained on Colab. Early stopping does not act before epoch 30.
- Phase 6 code: `scripts/evaluate.py` (k bands, M0 comparison, calibration, interpolation/ablation, fault detection, compact model, ONNX + verification, automatic model card, bundle zip), `eval/m0_batch.py`, `eval/trust.py`, `models/export.py`, `configs/train_m2_compact.yaml` (61,910 parameters). Dry run on pilot data done; ONNX vs PyTorch difference 2e-6.
- Tests: 140 passed.

## Phase 6 — 2026-10-19

- Locked test set (realistic contact, n = 3,715):
  - k median / p95: 0.05–0.5 → 0.83% / 2.60% · 0.5–5 → 0.89% / 3.17% · 5–30 → 2.07% / 10.36%.
  - M0 over all: 1.71% / 13.47%.
- Calibration: ln k 95% coverage 97%. The ablation model on its unseen band: 0.92% / 2.91%.
- Fault detection: thin sample 98%, drift jump 85%, air gap 53%. G_c range check and re-simulation (M0+) missing.
- Compact model 2 × 61,910 parameters, difference to the ensemble 0.27%. ONNX verified (2e-6).
- Model card: `models/m2_v1/MODEL_CARD.md`. ONNX and weights on Drive (outside git).

## Trust checks v2 (code) — 2026-10-19

- `eval/trust.py`: training-range check (ln k, ln α, ln G_c), ensemble spread for every target, shared 5% false-alarm budget, variance scale from val. Settings in `configs/trust.yaml`.
- `scripts/evaluate.py --trust-only`: val/test/test_faulty only; compares v1 with v2, writes `reports/phase6_trust/` and `phase6_trust_bundle.zip`. Labelled as the **second use** of the test sets.
- M0+ design: `docs/M0PLUS.md` (~50–75 s per measurement on one core).
- Dry run on pilot data done locally. Real results after the Colab run.

## Code quality, report and English — 2026-10-19

- Docstrings, no hard-coded constants, public helper names; full report `reports/REPORT_v1.md`.
- The whole repository was translated to English (comments, plot labels, reports, docs, model card, notebook). Phase 1–4 figures regenerated locally with English labels; Phase 6 figures follow with the next Colab evaluation run.

## Trust checks v2 (Colab results) — 2026-10-19

- Colab cell 15: the full Phase 6 evaluation reproduced every number exactly (same models and seeds), now with English figures; then the `--trust-only` run (**second use** of the test sets).
- Flagged, v1 → v2: air gap 52.8% → 74.3%, thin sample 97.9% → 99.3%, drift jump 84.9% → 92.1%, asymmetry > 10 4.2% → 5.2%; false alarm 4.7% → 5.7% on test (5.0% on val).
- Variance scale from val: ln k × 0.898, ln α × 0.589, ln G_c × 0.974; test coverage after scaling 69/95%, 71/93%, 68/96%.
- Unflagged faulty measurements: 229, median k error −28%. Asymmetric contact still needs M0+.
- Phase 6 figures (English) copied from Drive into `reports/phase6/`; trust v2 figures into `reports/phase6_trust/`.

## M0+ and closing the base model — 2026-10-19

- `eval/m0plus.py`: full inversion (`fit_m0plus`: ln k, ln α, both contacts; linear part: calibration scale, t_c, offset, drift) and screening (`m0plus_light`: 5 contact splits at M2's estimate). `scripts/run_m0plus.py`: trust v3 evaluation (resumable). `configs/m0plus.yaml`, trust v3 in `configs/trust.yaml`. Tests: `tests/test_m0plus.py`.
- Pilot development run (1,000 + 48 faulty, no test use): faults give χ²_ν 30–300× normal; asymmetric contact is not separable (see `reports/m0plus.md`).
- Decision: trust v2 stays shipped; M0+ kept as a tool; asymmetric contact to be handled by the measurement procedure. Repository tagged `v1.0-base`.
