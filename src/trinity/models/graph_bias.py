"""Netlist connectivity and padding as an additive attention bias.

The ``log1p`` b2b adjacency ``(B, N, N)`` times a learned scalar gate becomes a
``(B, 1, N, N)`` bias on the attention logits. Padded keys get ``PAD_BIAS`` added (after
the gate, whether or not the connectivity bias is enabled).
"""

import torch
import torch.nn as nn

PAD_BIAS = -1e4


class GraphBias(nn.Module):
    """Learned-gate connectivity bias plus padded-key masking."""

    def __init__(self, enabled: bool = True, init_gate: float = 1.0) -> None:
        super().__init__()
        self.enabled = enabled
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(
        self,
        adjacency: torch.Tensor | None,
        key_pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """``adjacency`` ``(B, N, N)``, ``key_pad_mask``
        ``(B, N)`` bool (True = real key).

        Returns the bias ``(B, 1, N, N)``, or ``None`` with neither a bias nor a mask.
        """
        bias = None
        if self.enabled and adjacency is not None:
            bias = (self.gate * adjacency).unsqueeze(1)
        if key_pad_mask is not None:
            pad = torch.where(key_pad_mask, 0.0, PAD_BIAS)[:, None, None, :]
            bias = pad if bias is None else bias + pad
        return bias
