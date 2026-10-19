"""Gaussian negative log-likelihood with the beta-NLL weighting (Seitzer et al., ICLR 2022).

per target: L = sg(var^beta) * (0.5 log var + 0.5 (y - mu)^2 / var)
beta = 0 is the plain Gaussian NLL; beta > 0 stops high-variance points from being ignored.
Targets are weighted (ln k 1, ln alpha 0.5, ln G_c 0.2) and averaged over the batch.
"""

from __future__ import annotations

import torch


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian NLL without the constant 0.5 log(2 pi)."""
    return 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))


def beta_nll(mean, logvar, y, beta: float = 0.5, weights=None) -> torch.Tensor:
    """Weighted beta-NLL averaged over the batch (see the module docstring)."""
    nll = gaussian_nll(mean, logvar, y)
    if beta > 0:
        nll = nll * torch.exp(beta * logvar).detach()
    if weights is not None:
        w = torch.as_tensor(weights, dtype=nll.dtype, device=nll.device)
        return (nll * w).sum(dim=-1).mean() / w.sum()
    return nll.mean()
