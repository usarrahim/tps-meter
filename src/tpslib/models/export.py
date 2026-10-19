"""Export an M2 ensemble (with its normalisers) to one ONNX graph and verify it against PyTorch.

Graph inputs : seq [B, C, 256] and scalars [B, S], the raw outputs of data.features.make_features
Graph outputs: mean [B, 3] and std [B, 3] in ln units (ln k, ln alpha, ln G_c; mixture of the members),
               member_mean [M, B, 3]
So a desktop/Android app only needs make_features + this file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class EnsembleWithNorm(nn.Module):
    """Ensemble members with their normalisers in one module (the ONNX export unit)."""
    def __init__(self, models: list[nn.Module], normalizers: list):
        super().__init__()
        self.members = nn.ModuleList(models)
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)
        for i, nm in enumerate(normalizers):
            self.register_buffer(f"sm{i}", f(nm.seq_mean)); self.register_buffer(f"ss{i}", f(nm.seq_std))
            self.register_buffer(f"cm{i}", f(nm.scal_mean)); self.register_buffer(f"cs{i}", f(nm.scal_std))
            self.register_buffer(f"ym{i}", f(nm.y_mean)); self.register_buffer(f"ys{i}", f(nm.y_std))

    def forward(self, seq, scalars):
        """Raw features -> mixture mean and std in ln units, plus the member means."""
        mus, vars_ = [], []
        for i, m in enumerate(self.members):
            b = lambda n: getattr(self, f"{n}{i}")
            s = (seq - b("sm")[None, :, None]) / b("ss")[None, :, None]
            c = (scalars - b("cm")[None, :]) / b("cs")[None, :]
            mu, lv = m(s, c)
            mus.append(mu * b("ys") + b("ym"))
            vars_.append(torch.exp(lv) * b("ys") ** 2)
        mu = torch.stack(mus)
        var = torch.stack(vars_).mean(0) + mu.var(0, unbiased=False)
        return mu.mean(0), torch.sqrt(var), mu


def export_onnx(model: EnsembleWithNorm, path: Path, n_channels: int, n_scalars: int, length: int = 256) -> Path:
    """Export to ONNX (opset 17, dynamic batch size); returns the file path."""
    model = model.cpu().eval()
    seq = torch.randn(2, n_channels, length)
    sc = torch.randn(2, n_scalars)
    torch.onnx.export(model, (seq, sc), str(path), input_names=["seq", "scalars"],
                      output_names=["mean", "std", "member_mean"], opset_version=17,
                      dynamic_axes={"seq": {0: "batch"}, "scalars": {0: "batch"}, "mean": {0: "batch"},
                                    "std": {0: "batch"}, "member_mean": {1: "batch"}}, dynamo=False)
    return Path(path)


def verify_onnx(model: EnsembleWithNorm, path: Path, seq: np.ndarray, scalars: np.ndarray) -> dict:
    """Max absolute difference between PyTorch and onnxruntime outputs on the given inputs."""
    import onnxruntime as ort

    model = model.cpu().eval()
    with torch.no_grad():
        ref = [t.numpy() for t in model(torch.from_numpy(seq), torch.from_numpy(scalars))]
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"seq": seq.astype(np.float32), "scalars": scalars.astype(np.float32)})
    return {name: float(np.max(np.abs(a - b))) for name, a, b in zip(("mean", "std", "member_mean"), ref, out)}
