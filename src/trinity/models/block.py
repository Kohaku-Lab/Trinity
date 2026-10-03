"""The denoiser transformer block: pre-norm residual, optionally adaLN-Zero modulated.

When ``modulated``, the block receives the per-layer modulation ``mod`` (six chunks:
shift / scale / gate for attention and MLP) produced by the time-conditioning strategy and
applies adaLN-Zero gated residuals; otherwise ``mod`` is ``None`` and the block is a plain
pre-norm residual block. An optional graph-mix step runs first.
"""

import torch
import torch.nn as nn

from trinity.registry import ATTENTION, GRAPH_MIX, MLP, NORM, build
from trinity.utils import modulate


class DenoiserBlock(nn.Module):
    """Pre-norm transformer block; applies adaLN-Zero modulation when handed one."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        *,
        norm,
        mlp,
        attn,
        mlp_ratio: float,
        qk_norm: bool,
        modulated: bool = True,
        graph_mix="none",
    ) -> None:
        super().__init__()
        self.modulated = modulated
        self.graph_mix = build(graph_mix, GRAPH_MIX, dim=dim)
        self.norm1 = build(norm, NORM, dim=dim)
        self.attn = build(
            attn, ATTENTION, dim=dim, heads=heads, head_dim=head_dim, qk_norm=qk_norm
        )
        self.norm2 = build(norm, NORM, dim=dim)
        self.mlp = build(mlp, MLP, dim=dim, ratio=mlp_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mod: torch.Tensor | None,
        bias: torch.Tensor | None,
        adj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.graph_mix(x, adj)
        if self.modulated:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=-1
            )
            x = x + gate_msa.unsqueeze(1) * self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa), bias
            )
            x = x + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm2(x), shift_mlp, scale_mlp)
            )
            return x
        x = x + self.attn(self.norm1(x), bias)
        x = x + self.mlp(self.norm2(x))
        return x
