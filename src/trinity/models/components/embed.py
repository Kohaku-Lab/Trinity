"""Input and timestep embedders of the block-token denoiser (one token per block).

``InputEmbed`` projects ``[z_t || features || graph_pe]`` to the model width;
``TimestepEmbedder`` is the DiT sinusoidal + MLP time embedding.
"""

import math

import torch
import torch.nn as nn


class InputEmbed(nn.Module):
    """Project ``[z_t || features || graph_pe]`` ``(B, N, *)`` to ``(B, N, dim)``.

    ``graph_pe_dim = 0`` leaves the graph PE out.
    """

    def __init__(
        self, latent_dim: int, feature_dim: int, dim: int, graph_pe_dim: int = 0
    ) -> None:
        super().__init__()
        self.graph_pe_dim = graph_pe_dim
        self.proj = nn.Linear(latent_dim + feature_dim + graph_pe_dim, dim)

    def forward(
        self,
        z_t: torch.Tensor,
        features: torch.Tensor,
        graph_pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [z_t, features]
        if self.graph_pe_dim > 0 and graph_pe is not None:
            parts.append(graph_pe)
        return self.proj(torch.cat(parts, dim=-1))


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding of ``t * time_scale`` followed by a 2-layer MLP."""

    def __init__(
        self,
        dim: int,
        freq_dim: int = 256,
        time_scale: float = 1000.0,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.time_scale = time_scale
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def _sinusoid(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = (t.float() * self.time_scale)[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoid(t).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)
