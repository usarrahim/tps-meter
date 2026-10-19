# Phase 5 report (interim) — M2 training code and pilot trial (2026-10-18)

> This report shows that **the code is ready**. The numbers come from the 1,000-sample pilot data (900 train / 100 val). They are not a measure of model performance. The real training is done once the 64k-sample generation on Colab is finished.

## Status

| | |
|---|---|
| Code | `models/resnet1d.py`, `models/losses.py`, `train/trainer.py`, `scripts/train_m2.py`, Colab cells 7–12 |
| Parameter count | **375,078** (4 blocks 32–64–128–128, kernel 5, GELU, 2 convolutions per block; as described, by the Phase 0 decision) |
| Tests | **131 passed** (6 new training tests: shapes, NLL minimum, β-NLL gradient, epoch sampler, end-to-end training + resume, ablation) |
| Pilot trial | 40 epochs, batch 32, local RTX 3050 Ti: ~1 s per epoch. k median **1.8%**, p95 **14%**; α median 9%; 95% interval coverage **96%** |

## Settings as implemented (`configs/train_m2.yaml`)

- **Model:** after global average pooling, a 128-neuron head joined with the 7 scalars. Outputs: mean and log-variance for ln k, ln α and ln G_c. The log-variance is softly bounded to [−14, 4].
- **Loss:** β-NLL (β = 0.5); target weights ln k 1, ln α 0.5, ln G_c 0.2.
- **Optimisation:**
  - AdamW, learning rate 1e-3, weight decay 1e-4.
  - 2-epoch warm-up + cosine (per step).
  - Gradient clipping 1.0; batch 256; at most 60 epochs.
  - Early stopping on val ln k NLL, patience 8.
- **Mixed precision:** CUDA only. bf16 on A100/L4, fp16 + GradScaler elsewhere.
- **Data:**
  - Measurement effects are redrawn every epoch in train. The epoch number is encoded in the sample index (`EpochSampler`), so persistent DataLoader workers also produce new noise every epoch.
  - Effects are fixed in val.
  - Normalisation from train only (5,000 samples).
  - For ensemble diversity, each seed also has a different noise seed.
- **Run folder** `runs/<tag>_m2_seed<n>/`:
  - Record: `config.yaml`, `run_info.json` (git commit, code hash, device, data files), `normalizer.json`.
  - Weights: `best.pt`, `last.pt`.
  - Monitoring: `metrics.csv`, `learning_curve.png`, `summary.json`.
  - **The same command resumes an interrupted run from the last epoch** (tested).
- **Ablation:** `--exclude-k-band 1.5 2.5` removes the band from train **and** val (tested).
- **Data size study:** `--max-train 10000/25000` (for the Phase 6 decision "is more data needed").

## Pilot observations

- With batch 256, 900 samples give only 3 steps per epoch; the first 5-epoch trial was not meaningful. With batch 32 and 40 epochs the model learned steadily (`runs/pilot-trial-b32_m2_seed1/learning_curve.png`). With 50k samples, batch 256 gives ~195 steps per epoch; no problem.
- The val ln k NLL was still falling at epoch 40: at this data size the model is data-limited.
- The β-NLL loss value flattens early (the β weight suppresses the scale). Monitoring and early stopping therefore use the NLL and error in physical units.

## Next steps (Colab, after generation)

1. Upload the new zip to Drive. **Do not upload before generation finishes**, because the changed code would make it reject the unfinished shards.
2. GPU runtime (L4 is enough) → cells 1, 2, 7 → 8 (5-epoch trial): let us review the learning curve together.
3. After approval, 9 (data size), 10 (5-seed ensemble), 11 (ablation).
