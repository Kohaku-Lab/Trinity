"""Time-conditioning strategies: how the timestep
embedding ``c (B, dim)`` enters the backbone.

* ``adaln`` -- DiT adaLN-Zero: one ``SiLU + Linear(dim, 6 dim)`` projection per block.
* ``adaln_shared`` -- one projection shared by every block; ``per_layer_affine`` adds a
  learned per-block ``(scale, shift)`` on the shared modulation.
* ``additive`` -- ``c`` added to every token once at the input.
* ``token`` -- ``c`` prepended as one extra token (removed
  before the head); the graph bias is padded to match.

Each strategy exposes ``enter(x, c, bias) -> (x, bias)``, ``modulation(c) -> list`` (one
entry per block, ``None`` for the unmodulated strategies) and ``exit(x) -> x``;
``owns_block_modulation`` says whether the blocks and the head use adaLN.
"""

import torch
import torch.nn as nn

from trinity.models.graph_bias import PAD_BIAS
from trinity.registry import TIME_COND


@TIME_COND.register("adaln")
class AdaLNCond(nn.Module):
    """DiT adaLN-Zero: one ``Linear(dim, 6*dim)``
    projection per block, gated residuals."""

    owns_block_modulation = True

    def __init__(self, dim: int, depth: int, **_unused) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim)) for _ in range(depth)
        )

    def enter(self, x, c, bias, key_pad_mask=None):
        return x, bias

    def modulation(self, c):
        return [p(c) for p in self.proj]

    def exit(self, x):
        return x


@TIME_COND.register("adaln_shared")
class SharedAdaLNCond(nn.Module):
    """Shared adaLN: one ``SiLU + Linear(dim, 6 dim)`` modulation reused by every block.

    ``per_layer_affine`` applies a learned per-block ``(scale, shift)`` (init 1, 0).
    """

    owns_block_modulation = True

    def __init__(
        self, dim: int, depth: int, per_layer_affine: bool = False, **_unused
    ) -> None:
        super().__init__()
        self.depth = depth
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        if per_layer_affine:
            self.layer_scale = nn.Parameter(torch.ones(depth, 6 * dim))
            self.layer_shift = nn.Parameter(torch.zeros(depth, 6 * dim))
            self.modulation = self._modulation_affine
        else:
            self.modulation = self._modulation_shared

    def enter(self, x, c, bias, key_pad_mask=None):
        return x, bias

    def _modulation_shared(self, c):
        return [self.proj(c)] * self.depth

    def _modulation_affine(self, c):
        m = self.proj(c)
        return [
            m * self.layer_scale[i] + self.layer_shift[i] for i in range(self.depth)
        ]

    def exit(self, x):
        return x


class _PlainResidualCond(nn.Module):
    """Base of the unmodulated strategies; ``modulation`` returns ``[None] * depth``."""

    owns_block_modulation = False

    def __init__(self, depth: int, **_unused) -> None:
        super().__init__()
        self._none_mods = [None] * depth

    def modulation(self, c):
        return self._none_mods


@TIME_COND.register("additive")
class AdditiveCond(_PlainResidualCond):
    """``x + c`` on every token, once before the blocks."""

    def enter(self, x, c, bias, key_pad_mask=None):
        return x + c.unsqueeze(1), bias

    def exit(self, x):
        return x


@TIME_COND.register("token")
class TokenCond(_PlainResidualCond):
    """The time embedding as an extra token at position 0, removed before the head.

    The bias ``(B, 1, N, N)`` is padded to ``(B, 1, N+1, N+1)`` (no edges for the time
    token; as a query it keeps the padded-key mask).
    """

    def enter(self, x, c, bias, key_pad_mask=None):
        x = torch.cat([c.unsqueeze(1), x], dim=1)
        if bias is not None:
            bias = nn.functional.pad(bias, (1, 0, 1, 0))
            if key_pad_mask is not None:
                bias[:, :, 0, 1:] = torch.where(key_pad_mask, 0.0, PAD_BIAS).unsqueeze(
                    1
                )
        return x, bias

    def exit(self, x):
        return x[:, 1:]
