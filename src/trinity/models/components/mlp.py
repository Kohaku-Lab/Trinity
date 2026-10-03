"""Feed-forward blocks of the transformer.

``gelu`` -- the DiT MLP (tanh-approximate GELU); ``swiglu`` -- SwiGLU with hidden width
``2/3 * ratio * dim`` rounded up to ``multiple_of``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.registry import MLP


@MLP.register("gelu")
class GELUMLP(nn.Module):
    """Two-layer GELU MLP."""

    def __init__(self, dim: int, ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


@MLP.register("swiglu")
class SwiGLU(nn.Module):
    """SwiGLU MLP; gate and value share one input projection."""

    def __init__(self, dim: int, ratio: float = 4.0, multiple_of: int = 64) -> None:
        super().__init__()
        hidden = int(2 * ratio * dim / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * value)
