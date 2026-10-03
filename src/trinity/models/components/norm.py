"""Normalization layers over ``(B, N, D)``: ``layernorm`` and ``rmsnorm``."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.registry import NORM


@NORM.register("layernorm")
class LayerNorm(nn.Module):
    """LayerNorm over the last dim (non-affine by default)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


@NORM.register("rmsnorm")
class RMSNorm(nn.Module):
    """RMSNorm over the last dim (affine by default)."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
