"""M2 training loop: AdamW + warmup/cosine, beta-NLL, AMP on CUDA, early stopping, resumable runs.

Every run lives in its own folder <runs root>/<name>/ with
  config.yaml, run_info.json (git commit, code hash, device, data files), normalizer.json,
  best.pt (weights of the best val epoch), last.pt (full state for resuming), metrics.csv,
  learning_curve.png.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

from tpslib.data.dataset import Normalizer, TPSDataset, select_indices
from tpslib.data.generate import _git_commit, code_hash
from tpslib.models.losses import beta_nll, gaussian_nll
from tpslib.models.resnet1d import build_model, count_parameters
from tpslib.runtime import describe_device, get_device


class EpochSampler(Sampler):
    """Yields epoch * n + i so that dataset workers know the epoch (new augmentation each epoch)."""

    def __init__(self, n: int, shuffle: bool, seed: int):
        self.n, self.shuffle, self.seed, self.epoch = n, shuffle, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        order = (np.random.default_rng([self.seed, self.epoch]).permutation(self.n) if self.shuffle
                 else np.arange(self.n))
        return iter((self.epoch * self.n + order).tolist())


def _worker_init(_):
    torch.set_num_threads(1)


def target_weights(cfg: dict) -> list[float]:
    w = cfg["loss"]["weights"]
    return [float(w[t]) for t in cfg["model"]["targets"]]


@torch.no_grad()
def evaluate(model, loader, normalizer: Normalizer, device, beta: float = 0.5, weights=None) -> dict:
    """Validation metrics: beta-NLL (normalised units, like the train loss) and, in physical ln units,
    NLL per target, relative errors and interval coverage."""
    model.eval()
    mus, lvs, ys = [], [], []
    for seq, scal, y in loader:
        mu, lv = model(seq.to(device, non_blocking=True), scal.to(device, non_blocking=True))
        mus.append(mu.float().cpu()); lvs.append(lv.float().cpu()); ys.append(y)
    mu, lv, y = torch.cat(mus), torch.cat(lvs), torch.cat(ys)
    val_loss = float(beta_nll(mu, lv, y, beta=beta, weights=weights))
    ym = torch.as_tensor(normalizer.y_mean, dtype=torch.float32)
    ys_ = torch.as_tensor(normalizer.y_std, dtype=torch.float32)
    mu_p, y_p = mu * ys_ + ym, y * ys_ + ym
    lv_p = lv + 2 * torch.log(ys_)
    nll = gaussian_nll(mu_p, lv_p, y_p).mean(dim=0) + 0.5 * math.log(2 * math.pi)
    rel = torch.exp(mu_p - y_p) - 1.0
    z = (mu_p - y_p).abs() / torch.exp(0.5 * lv_p)
    out = {"val_loss": val_loss}
    for i, name in enumerate(["ln_k", "ln_alpha", "ln_G_c"][: mu.shape[1]]):
        r = rel[:, i].abs().numpy()
        out[f"val_nll_{name}"] = float(nll[i])
        out[f"val_med_rel_err_{name}"] = float(np.median(r))
        out[f"val_p95_rel_err_{name}"] = float(np.percentile(r, 95))
        out[f"val_cov68_{name}"] = float((z[:, i] <= 1.0).float().mean())
        out[f"val_cov95_{name}"] = float((z[:, i] <= 1.96).float().mean())
    model.train()
    return out


def _plot_curves(rows: list[dict], path: Path) -> None:
    from tpslib.eval.report import SERIES, plt, save, style

    ep = [r["epoch"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(ep, [r["train_loss"] for r in rows], "-o", color=SERIES[0], ms=3, label="train (β-NLL)")
    axs[0].plot(ep, [r["val_loss"] for r in rows], "-s", color=SERIES[1], ms=3, label="val (β-NLL)")
    axs[0].set_xlabel("epok"); axs[0].legend(frameon=False); style(axs[0], "Kayıp (normalize birimler)")
    for i, t in enumerate(["ln_k", "ln_alpha", "ln_G_c"]):
        axs[1].plot(ep, [r[f"val_nll_{t}"] for r in rows], "-o", ms=3, color=SERIES[i], label=t)
    axs[1].set_xlabel("epok"); axs[1].legend(frameon=False); style(axs[1], "Val NLL (ln birimleri)")
    for i, t in enumerate(["ln_k", "ln_alpha"]):
        axs[2].plot(ep, [100 * r[f"val_med_rel_err_{t}"] for r in rows], "-o", ms=3, color=SERIES[i],
                    label=f"{t[3:]} medyan")
        axs[2].plot(ep, [100 * r[f"val_p95_rel_err_{t}"] for r in rows], "--", color=SERIES[i],
                    label=f"{t[3:]} %95")
    axs[2].set_yscale("log"); axs[2].set_xlabel("epok"); axs[2].set_ylabel("bağıl hata [%]")
    axs[2].legend(frameon=False, fontsize=8); style(axs[2], "Val bağıl hata")
    save(fig, path)


def train(cfg: dict, data_dir: Path, run_dir: Path, seed: int, resume: bool = True,
          train_file: str | None = None, val_file: str | None = None, dev_split: float | None = None,
          max_epochs: int | None = None, num_workers: int | None = None, log=print) -> dict:
    """Train one M2 model. Returns the summary (best epoch metrics)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()
    dc, tc, oc = cfg["data"], cfg["train"], cfg["optim"]
    max_epochs = int(max_epochs or tc["max_epochs"])
    nw = int(tc["num_workers"] if num_workers is None else num_workers)

    # --- data ---------------------------------------------------------------------------------
    train_path = data_dir / (train_file or dc["train"])
    val_path = data_dir / (val_file or dc["val"])
    aug_seed = 1_000_003 * seed + 17
    tr = TPSDataset(train_path, train=True, base_seed=aug_seed)
    if dev_split is not None:          # development only: split one file into train/val parts
        n_tr = int(len(tr) * dev_split)
        perm = np.random.default_rng(12345).permutation(tr.indices)
        tr.indices = np.sort(perm[:n_tr])
        va = TPSDataset(train_path, train=False, base_seed=2002, indices=np.sort(perm[n_tr:]))
    else:
        va = TPSDataset(val_path, train=False, base_seed=2002)
    if dc.get("exclude_k_band") or dc.get("max_contact_asymmetry"):
        for ds in (tr, va):
            keep = select_indices(ds.params, exclude_k_band=dc.get("exclude_k_band"),
                                  max_contact_asymmetry=dc.get("max_contact_asymmetry"))
            ds.indices = np.intersect1d(ds.indices, keep)
    if dc.get("max_train"):
        tr.indices = np.sort(np.random.default_rng(seed).permutation(tr.indices)[: int(dc["max_train"])])

    norm_path = run_dir / "normalizer.json"
    if norm_path.exists():
        normalizer = Normalizer.load(norm_path)
    else:
        t0 = time.perf_counter()
        normalizer = Normalizer.fit(tr, n=int(dc["normalizer_samples"]), seed=seed)
        normalizer.save(norm_path)
        log(f"normalizer fitted on {normalizer.n_samples} train samples in {time.perf_counter() - t0:.0f} s")
    tr.normalizer = va.normalizer = normalizer

    sampler = EpochSampler(len(tr), shuffle=True, seed=seed)
    bs = int(tc["batch_size"])
    kw = dict(num_workers=nw, pin_memory=device.type == "cuda", worker_init_fn=_worker_init,
              persistent_workers=nw > 0)
    dl_tr = DataLoader(tr, batch_size=bs, sampler=sampler, drop_last=len(tr) > bs, **kw)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, **kw)

    # --- model / optimisation -----------------------------------------------------------------
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(oc["lr"]), weight_decay=float(oc["weight_decay"]))
    steps_per_epoch = max(1, len(dl_tr))
    warm = int(oc["warmup_epochs"]) * steps_per_epoch
    total = max_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = tc.get("amp") == "cuda_only" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    weights = target_weights(cfg)
    beta = float(cfg["loss"]["beta"])
    es_metric, patience = tc["early_stopping"]["metric"], int(tc["early_stopping"]["patience"])
    # Early stopping only after min_epochs: during the high-LR phase the val metric is too noisy and
    # patience 8 stopped a 50k run at epoch 27 before the cosine decay (DECISIONS.md).
    min_epochs = int(tc["early_stopping"].get("min_epochs", 0))

    # --- run bookkeeping / resume ---------------------------------------------------------------
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    info = {"seed": seed, "device": describe_device(device), "n_params": count_parameters(model),
            "git": _git_commit(), "code_hash": code_hash(), "train_file": str(train_path),
            "val_file": str(val_path) if dev_split is None else f"{train_path} (dev split {dev_split})",
            "n_train": len(tr), "n_val": len(va), "amp": bool(use_amp), "amp_dtype": str(amp_dtype),
            "torch": torch.__version__, "num_workers": nw}
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    rows, start_epoch, best, bad = [], 0, math.inf, 0
    last = run_dir / "last.pt"
    if resume and last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start_epoch, best, bad, rows = st["epoch"] + 1, st["best"], st["bad"], st["rows"]
        if start_epoch <= min_epochs:  # runs saved before the min_epochs rule: restart the count
            bad = 0
        log(f"resumed from epoch {st['epoch']}")
    log(f"params={info['n_params']}, device={info['device']}, train={len(tr)}, val={len(va)}, amp={use_amp}")

    for epoch in range(start_epoch, max_epochs):
        if bad >= patience and epoch >= min_epochs:
            break
        t0 = time.perf_counter()
        sampler.set_epoch(epoch)
        model.train()
        tot, nb = 0.0, 0
        for seq, scal, y in dl_tr:
            seq, scal, y = (x.to(device, non_blocking=True) for x in (seq, scal, y))
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                mu, lv = model(seq, scal)
            loss = beta_nll(mu.float(), lv.float(), y, beta=beta, weights=weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(oc["grad_clip"]))
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        t_train = time.perf_counter() - t0
        m = evaluate(model, dl_va, normalizer, device, beta=beta, weights=weights)
        row = {"epoch": epoch, "train_loss": tot / max(nb, 1), "lr": sched.get_last_lr()[0],
               "time_train_s": t_train, "time_total_s": time.perf_counter() - t0, **m}
        rows.append(row)
        improved = row[es_metric] < best - 1e-6
        if improved:
            best, bad = row[es_metric], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "metrics": row}, run_dir / "best.pt")
        elif epoch >= min_epochs:      # patience counts only after min_epochs
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad": bad, "rows": rows},
                   last.with_suffix(".tmp"))
        os.replace(last.with_suffix(".tmp"), last)
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        _plot_curves(rows, run_dir / "learning_curve.png")
        log(f"ep {epoch:3d} train {row['train_loss']:.4f} val {row['val_loss']:.4f} "
            f"k med {100 * m['val_med_rel_err_ln_k']:.2f}% p95 {100 * m['val_p95_rel_err_ln_k']:.2f}% "
            f"a med {100 * m['val_med_rel_err_ln_alpha']:.2f}% cov95(k) {m['val_cov95_ln_k']:.2f} "
            f"{'*' if improved else ' '} {row['time_total_s']:.0f}s")
    best_row = min(rows, key=lambda r: r[es_metric]) if rows else {}
    summary = {"best_epoch": best_row.get("epoch"), "epochs_run": len(rows), **info, "best": best_row}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
