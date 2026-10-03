"""Differentiable torch decode of the latent
``z = (cx/s, cy/s, rho)`` to ``(x, y, w, h)``.

The torch counterpart of :func:`trinity.floorplan.parameterize.z_to_xywh`:
``w = sqrt(a) e^{rho/2}``, ``h = sqrt(a) e^{-rho/2}``, so ``w * h == a``.
"""

import torch

from trinity.floorplan.parameterize import RHO_CLAMP


def z_to_xywh(
    z: torch.Tensor, area_targets: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """``z`` ``(B, N, 3)``, ``area_targets`` ``(B,
    N)``, ``scale`` ``(B,)`` -> ``(B, N, 4)``."""
    rho = z[..., 2].clamp(-RHO_CLAMP, RHO_CLAMP)
    root_a = area_targets.clamp_min(0).sqrt()
    w = root_a * torch.exp(rho / 2)
    h = root_a * torch.exp(-rho / 2)
    s = scale.unsqueeze(-1)
    cx = z[..., 0] * s
    cy = z[..., 1] * s
    x = cx - w / 2
    y = cy - h / 2
    return torch.stack([x, y, w, h], dim=-1)
