"""Graph mix (the "info mover"): a message-passing step along netlist edges per block.

``mp`` computes ``x + gate * W (A_hat x)`` with ``A_hat`` the normalized adjacency and a
zero-initialized scalar gate; ``none`` is the identity.
"""

import torch
import torch.nn as nn

from trinity.registry import GRAPH_MIX


@GRAPH_MIX.register("none")
class NoGraphMix(nn.Module):
    """The identity."""

    def __init__(self, **_unused) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        return x


@GRAPH_MIX.register("mp")
class MessagePassingMix(nn.Module):
    """``x + gate * W (A_hat X)``. ``normalize``: ``row``
    (mean of neighbours) | ``sym`` | ``none``."""

    def __init__(self, dim: int, normalize: str = "row", **_unused) -> None:
        super().__init__()
        self.normalize = normalize
        self.proj = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def _norm(self, adj: torch.Tensor) -> torch.Tensor:
        """``A_hat``: ``D^-1 A`` (row), ``D^-1/2 A D^-1/2`` (sym) or ``A`` (none)."""
        if self.normalize == "none":
            return adj
        deg = adj.sum(-1, keepdim=True).clamp_min(1e-6)
        if self.normalize == "row":
            return adj / deg
        d = deg.rsqrt()
        return d * adj * d.transpose(-1, -2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if adj is None:
            return x
        return x + self.gate * self.proj(torch.bmm(self._norm(adj).to(x.dtype), x))
