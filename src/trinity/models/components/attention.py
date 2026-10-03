"""Self-attention over block tokens, with an optional additive graph bias.

Registered variants:

* ``sdpa`` -- fused attention, the bias ignored;
* ``sdpa_graph`` -- fused attention with the bias as an additive ``attn_mask``;
* ``sdpa_graph_heads`` -- the first ``n_graph_heads`` heads attend by the bias alone;
* ``flex_graph`` -- FlexAttention with the bias added in ``score_mod``;
* ``naive_graph`` -- the explicit ``softmax(QK^T / sqrt(d) + bias)``.

Every forward takes ``(x, bias)`` with ``bias`` ``(B, 1, N, N)`` or ``None``. With
``qk_norm``, a non-learnable RMSNorm over the head dim is applied to queries and keys.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.registry import ATTENTION

try:
    from torch.nn.attention.flex_attention import flex_attention as _flex_attention

    _HAS_FLEX = True
except ImportError:
    _HAS_FLEX = False


class _AttnBase(nn.Module):
    """Fused QKV + output projection + optional QK-norm; subclasses define the op."""

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim or (dim // heads)
        inner = self.heads * self.head_dim
        self.qkv = nn.Linear(dim, 3 * inner, bias=qkv_bias)
        self.proj = nn.Linear(inner, dim)
        self.qk_norm = qk_norm

    def _qkv(self, x: torch.Tensor):
        """Queries, keys and values, each ``(B, H, N, head_dim)``."""
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,))
            k = F.rms_norm(k, (self.head_dim,))
        return q, k, v

    def _out(self, out: torch.Tensor, b: int, n: int) -> torch.Tensor:
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.proj(out)


@ATTENTION.register("sdpa")
class SDPAAttention(_AttnBase):
    """Fused self-attention; ``bias`` is ignored."""

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        out = F.scaled_dot_product_attention(q, k, v)
        return self._out(out, b, n)


@ATTENTION.register("sdpa_graph")
class SDPAGraphAttention(_AttnBase):
    """Fused self-attention with the graph bias passed as an additive ``attn_mask``."""

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self._out(out, b, n)


@ATTENTION.register("flex_graph")
class FlexGraphAttention(_AttnBase):
    """FlexAttention with the graph bias added in a
    ``score_mod`` closure (fused under compile)."""

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        if bias is None:
            out = _flex_attention(q, k, v)
        else:

            def score_mod(score, batch, head, q_idx, kv_idx):
                return score + bias[batch, 0, q_idx, kv_idx]

            out = _flex_attention(q, k, v, score_mod=score_mod)
        return self._out(out, b, n)


@ATTENTION.register("sdpa_graph_heads")
class SDPAGraphHeadsAttention(_AttnBase):
    """Head-split graph attention: ``n_graph_heads`` heads attend by the netlist alone.

    In the first ``n_graph_heads`` heads the queries are zeroed and the logits are
    ``softplus(scale) * bias + shift`` (learned per head); the other heads are
    ``sdpa_graph``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int | None = None,
        qk_norm: bool = False,
        qkv_bias: bool = True,
        n_graph_heads: int = 2,
    ) -> None:
        super().__init__(dim, heads, head_dim, qk_norm, qkv_bias)
        if not 0 <= n_graph_heads <= heads:
            raise ValueError(
                f"n_graph_heads {n_graph_heads} must be in [0, heads={heads}]"
            )
        self.n_graph_heads = n_graph_heads
        # softplus(0.5413) = 1.
        self.logit_scale = nn.Parameter(torch.full((max(n_graph_heads, 1),), 0.5413))
        self.logit_shift = nn.Parameter(torch.zeros(max(n_graph_heads, 1)))

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        gh = self.n_graph_heads
        if bias is None or gh == 0:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            return self._out(out, b, n)
        mask = bias.expand(b, self.heads, n, n).clone()
        scale = F.softplus(self.logit_scale).view(1, gh, 1, 1)
        shift = self.logit_shift.view(1, gh, 1, 1)
        mask[:, :gh] = scale * bias + shift
        q = torch.cat([torch.zeros_like(q[:, :gh]), q[:, gh:]], dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._out(out, b, n)


@ATTENTION.register("naive_graph")
class NaiveGraphAttention(_AttnBase):
    """The explicit ``softmax(QK^T / sqrt(d) + bias) V``."""

    def forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = self._qkv(x)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        out = torch.matmul(F.softmax(logits, dim=-1), v)
        return self._out(out, b, n)
