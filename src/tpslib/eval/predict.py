"""Load trained M2 runs and predict on a dataset (physical ln units, with predictive std)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tpslib.data.dataset import Normalizer, TPSDataset
from tpslib.models.resnet1d import build_model
from tpslib.runtime import get_device


def load_run(run_dir, device=None):
    """Model (best epoch, eval mode), normaliser and config of one run folder."""
    run_dir = Path(run_dir)
    device = device or get_device()
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, Normalizer.load(run_dir / "normalizer.json"), ck["cfg"]


@torch.no_grad()
def predict(model, normalizer: Normalizer, ds: TPSDataset, device=None, batch_size: int = 512,
            num_workers: int = 0) -> dict:
    """Predictions for every item of `ds` (eval mode effects). Returns ln-unit mean, std and truth."""
    device = device or next(model.parameters()).device
    ds.normalizer = normalizer
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    persistent_workers=False, pin_memory=device.type == "cuda")
    mus, lvs, ys = [], [], []
    for seq, scal, y in dl:
        mu, lv = model(seq.to(device), scal.to(device))
        mus.append(mu.float().cpu().numpy()); lvs.append(lv.float().cpu().numpy()); ys.append(y.numpy())
    mu, lv, y = np.concatenate(mus), np.concatenate(lvs), np.concatenate(ys)
    ym, ys_ = np.asarray(normalizer.y_mean), np.asarray(normalizer.y_std)
    return {"mean": mu * ys_ + ym, "std": np.exp(0.5 * lv) * ys_, "true": y * ys_ + ym,
            "index": np.asarray(ds.indices)}


@torch.no_grad()
def predict_members(members: list[tuple], ds: TPSDataset, device=None, batch_size: int = 512,
                    num_workers: int = 0) -> list[dict]:
    """Like `predict` for several (model, normalizer) pairs, computing the features only once."""
    device = device or next(members[0][0].parameters()).device
    ds.normalizer = None
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    persistent_workers=False, pin_memory=device.type == "cuda")
    acc = [{"mean": [], "lv": []} for _ in members]
    ys = []
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    for seq, scal, y in dl:
        seq, scal = seq.to(device), scal.to(device)
        ys.append(y.numpy())
        for a, (model, nm) in zip(acc, members):
            s = (seq - t(nm.seq_mean)[None, :, None]) / t(nm.seq_std)[None, :, None]
            c = (scal - t(nm.scal_mean)) / t(nm.scal_std)
            mu, lv = model(s, c)
            a["mean"].append(mu.float().cpu().numpy()); a["lv"].append(lv.float().cpu().numpy())
    y = np.concatenate(ys)
    out = []
    for a, (_, nm) in zip(acc, members):
        ym, ys_ = np.asarray(nm.y_mean), np.asarray(nm.y_std)
        out.append({"mean": np.concatenate(a["mean"]) * ys_ + ym,
                    "std": np.exp(0.5 * np.concatenate(a["lv"])) * ys_,
                    "true": y.astype(np.float64), "index": np.asarray(ds.indices)})
    return out


def ensemble(preds: list[dict]) -> dict:
    """Gaussian mixture of ensemble members: mean of means, total variance = aleatoric + epistemic."""
    mu = np.stack([p["mean"] for p in preds])
    var = np.stack([p["std"] ** 2 for p in preds])
    m = mu.mean(axis=0)
    return {"mean": m, "std": np.sqrt(var.mean(axis=0) + mu.var(axis=0)), "std_epistemic": mu.std(axis=0),
            "true": preds[0]["true"], "index": preds[0]["index"]}
