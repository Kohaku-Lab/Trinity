"""The direct regressor: the diffusion backbone run as a deterministic predictor.

Wraps :class:`trinity.models.SetTransformerDenoiser` and drives it with a constant input
``z_t = 0`` and time ``t = 0``, so the prediction depends on the conditioning only. The
backbone's ``latent_dim`` is the head's ``out_dim`` (3 for ``z``, 4 for ``xywh``).
Anchor values reach the network as conditioning feature columns; no clamp is applied.
"""

import torch
import torch.nn as nn

from trinity.models import DenoiserCond, SetTransformerDenoiser
from trinity.models.presets import DenoiserArchConfig


class DirectRegressor(nn.Module):
    """A set-transformer regressor: ``cond ->
    per-block layout prediction`` (no diffusion)."""

    def __init__(self, arch: DenoiserArchConfig, head) -> None:
        super().__init__()
        arch.latent_dim = head.out_dim
        self.arch = arch
        self.head = head
        self.backbone = SetTransformerDenoiser(arch)

    @property
    def blocks(self) -> nn.ModuleList:
        """The inner denoiser's block list (the per-module compile target)."""
        return self.backbone.blocks

    def _const_inputs(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The constant ``(z_t = 0, t = 0)`` driving inputs for one batch."""
        b, n = features.shape[0], features.shape[1]
        z_t = features.new_zeros(b, n, self.arch.latent_dim)
        t = features.new_zeros(b)
        return z_t, t

    def forward(self, cond: DenoiserCond) -> torch.Tensor:
        """Predict the per-block layout ``(B, N,
        out_dim)`` in the head's space from ``cond``."""
        z_t, t = self._const_inputs(cond.features)
        return self.backbone(z_t, t, cond)
