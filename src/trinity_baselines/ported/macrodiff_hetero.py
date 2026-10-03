"""MacroDiff+ ``MacroPlacer`` denoiser, ported
to the ``(z_t, t, cond) -> out`` interface.

Source: https://github.com/jhy00n/MacroDiff-plus (MIT; licenses: NOTICE). Copied from
``models/{graph,trans,model}.py`` at the training configuration (``train.py``: hidden
64, 5 hetero GATv2 layers, 4 heads, edge_dim 2; ``TransConv`` with 32 base channels,
multipliers (1, 2, 4, 8), attention at levels 1 and 2; T = 200). The forward keeps their
epsilon composition ``alpha * (dHPWL/dx)^T net_head + beta * cell_head``
(``diffuser.py``, ``noise='full'``, alpha = beta = 1). Changes at the batch boundary:

* their cell / net bipartite graph is built from the dense b2b adjacency: one net node
  per nonzero pair (a 2-pin net), edges cell->net and net->cell, edge attribute
  ``(weight, 1)``; pins enter through the per-block conditioning columns rather than as
  fixed IO cells;
* cell features are the 3-D latent ``(cx, cy, rho)`` + the 19-D conditioning (theirs:
  position, size, is_macro); net features are ``(net length at z_t, weight, degree)``
  (theirs: length delta, degree); the transformer's cross-attention context is the 19-D
  conditioning (theirs: the macro sizes);
* the HPWL derivative of a 2-pin net w.r.t. an endpoint is ``sign(p_i - p_j)`` per axis,
  so the net head's contribution to a block is the signed sum over its nets; it enters
  the first two output channels only; the integer timestep embedding receives ``t *
  200``.

Registered as ``BASELINE_BACKBONE["macrodiff_hetero"]``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.models import DenoiserCond
from trinity_baselines.registry import BASELINE_BACKBONE


class SinusoidalPositionEmbeddings(nn.Module):
    """Their timestep embedding ``(M,) -> (M, dim)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.arange(half, dtype=torch.float32)
                * -(math.log(10000) / (half - 1))
            ),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        e = t.float()[:, None] * self.freqs[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)


def _gatv2(hidden: int, edge_dim: int, heads: int, dropout: float):
    from torch_geometric.nn import GATv2Conv

    return GATv2Conv(
        hidden,
        hidden,
        edge_dim=edge_dim,
        heads=heads,
        dropout=dropout,
        add_self_loops=False,
        concat=False,
    )


class GraphConv(nn.Module):
    """Their hetero GNN: per layer a cell->net and a net->cell GATv2 (mean over relation
    types), residual, LayerNorm, ELU, with the time embedding added to both node
    types."""

    def __init__(
        self,
        in_cell,
        in_net,
        hidden,
        edge_dim,
        time_emb_dim,
        num_layers,
        num_heads,
        dropout,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, hidden),
        )
        self.cell_in = nn.Linear(in_cell, hidden)
        self.net_in = nn.Linear(in_net, hidden)
        self.cell_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_layers + 1)]
        )
        self.net_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(num_layers + 1)]
        )
        self.c2n = nn.ModuleList(
            [_gatv2(hidden, edge_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.n2c = nn.ModuleList(
            [_gatv2(hidden, edge_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x_cell, x_net, t_cell, t_net, ei_c2n, ei_n2c, edge_attr):
        te_cell = self.time_mlp(self.time_embedder(t_cell))
        te_net = self.time_mlp(self.time_embedder(t_net))
        c = F.dropout(
            F.elu(self.cell_norms[0](self.cell_in(x_cell))), self.dropout, self.training
        )
        n = F.dropout(
            F.elu(self.net_norms[0](self.net_in(x_net))), self.dropout, self.training
        )
        for i, (c2n, n2c) in enumerate(zip(self.c2n, self.n2c, strict=True)):
            c_in, n_in = c + te_cell, n + te_net
            n_new = c2n((c_in, n_in), ei_c2n, edge_attr) + n
            c_new = n2c((n_in, c_in), ei_n2c, edge_attr) + c
            c = F.dropout(
                F.elu(self.cell_norms[i + 1](c_new)), self.dropout, self.training
            )
            n = F.dropout(
                F.elu(self.net_norms[i + 1](n_new)), self.dropout, self.training
            )
        return c, n


class CrossAttention(nn.Module):
    """Their multi-head attention ``(B, T, Dq) x
    (B, S, Dc) -> (B, T, Dq)`` with a key mask."""

    def __init__(self, query_dim, context_dim, heads, dim_head, dropout) -> None:
        super().__init__()
        inner = dim_head * heads
        self.heads, self.scale = heads, dim_head**-0.5
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_k = nn.Linear(context_dim, inner, bias=False)
        self.to_v = nn.Linear(context_dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        q, k, v = (
            t.reshape(*t.shape[:-1], self.heads, -1).permute(0, 2, 1, 3)
            for t in (q, k, v)
        )
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if mask is not None:
            sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(sim.dtype).max)
        out = (
            torch.einsum("bhij,bhjd->bhid", sim.softmax(dim=-1), v)
            .permute(0, 2, 1, 3)
            .reshape(*x.shape[:-1], -1)
        )
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Their ``norm(x) + attn`` then ``norm(x) + ff`` block."""

    def __init__(self, dim, context_dim, heads, dim_head, dropout) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, context, mask):
        x = self.norm1(x) + self.attn(x, context, mask)
        return self.norm2(x) + self.ff(x)


class ResidualBlock(nn.Module):
    """Their per-token residual MLP with an additive time embedding."""

    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1) -> None:
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_emb_dim, out_ch), nn.GELU())
        self.layer1, self.layer2, self.layer_out = (
            nn.Linear(in_ch, out_ch),
            nn.Linear(out_ch, out_ch),
            nn.Linear(out_ch, out_ch),
        )
        self.norm1, self.norm2 = nn.LayerNorm(in_ch), nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb):
        h = F.gelu(self.layer1(self.norm1(x))) + self.time_mlp(time_emb).unsqueeze(1)
        h = self.layer_out(self.dropout(F.gelu(self.layer2(self.norm2(h)))))
        return h + self.skip(x)


class TransConv(nn.Module):
    """Their token U-Net: Linear down / up over channels with skip concatenation, self-
    and cross-attention at the chosen levels and in the middle, conditioned on per-token
    context.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        model_channels,
        condition_channels,
        channel_mults,
        attention_levels,
        num_heads,
        dropout,
    ) -> None:
        super().__init__()
        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.condition_proj = nn.Linear(condition_channels, model_channels)
        chans = [model_channels]
        self.down_res, self.down_lin, self.down_self, self.down_cross = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        for i, mult in enumerate(channel_mults):
            out = model_channels * mult
            self.down_res.append(ResidualBlock(chans[-1], chans[-1], time_emb_dim))
            self.down_lin.append(nn.Linear(chans[-1], out))
            chans.append(out)
            attn = i in attention_levels
            self.down_self.append(
                CrossAttentionBlock(out, out, num_heads, out // num_heads, dropout)
                if attn
                else nn.Identity()
            )
            self.down_cross.append(
                CrossAttentionBlock(
                    out, model_channels, num_heads, out // num_heads, dropout
                )
                if attn
                else nn.Identity()
            )
        mid = chans[-1]
        self.mid1 = ResidualBlock(mid, mid, time_emb_dim)
        self.mid_self = CrossAttentionBlock(
            mid, mid, num_heads, mid // num_heads, dropout
        )
        self.mid_cross = CrossAttentionBlock(
            mid, model_channels, num_heads, mid // num_heads, dropout
        )
        self.mid2 = ResidualBlock(mid, mid, time_emb_dim)
        self.up_lin, self.up_res, self.up_self, self.up_cross = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )
        for i, mult in reversed(list(enumerate(channel_mults))):
            out = model_channels * mult
            self.up_lin.append(
                nn.Sequential(nn.Linear(chans[-1], out), nn.LayerNorm(out))
            )
            skip_ch = out if i == 0 else out // 2
            self.up_res.append(ResidualBlock(out + skip_ch, out, time_emb_dim))
            chans.append(out)
            attn = i in attention_levels
            self.up_self.append(
                CrossAttentionBlock(out, out, num_heads, out // num_heads, dropout)
                if attn
                else nn.Identity()
            )
            self.up_cross.append(
                CrossAttentionBlock(
                    out, model_channels, num_heads, out // num_heads, dropout
                )
                if attn
                else nn.Identity()
            )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(chans[-1]), nn.Linear(chans[-1], out_channels)
        )

    @staticmethod
    def _attn(block, h, context, mask):
        return block(h, context, mask) if isinstance(block, CrossAttentionBlock) else h

    def forward(self, x, t, condition, mask):
        te = self.time_mlp(self.time_embedder(t))
        h = self.input_proj(x)
        cond = self.condition_proj(condition)
        skips = []
        for res, lin, sa, ca in zip(
            self.down_res, self.down_lin, self.down_self, self.down_cross, strict=True
        ):
            skip = res(h, te)
            skips.append(skip)
            h = self._attn(ca, self._attn(sa, lin(skip), None, mask), cond, mask)
        h = self.mid2(
            self.mid_cross(self.mid_self(self.mid1(h, te), None, mask), cond, mask), te
        )
        for lin, res, sa, ca in zip(
            self.up_lin, self.up_res, self.up_self, self.up_cross, strict=True
        ):
            h = res(torch.cat([lin(h), skips.pop()], dim=-1), te)
            h = self._attn(ca, self._attn(sa, h, None, mask), cond, mask)
        return self.output_proj(h)


@BASELINE_BACKBONE.register("macrodiff_hetero")
class MacroDiffHetero(nn.Module):
    """``MacroPlacer`` (hetero GNN -> token U-Net) with
    the HPWL-composed epsilon, over the conditioning."""

    def __init__(
        self,
        latent_dim: int = 3,
        feature_dim: int = 19,
        hidden_channels: int = 64,
        num_graph_layers: int = 5,
        num_heads: int = 4,
        edge_dim: int = 2,
        model_channels: int = 32,
        channel_mults: tuple = (1, 2, 4, 8),
        attention_levels: tuple = (1, 2),
        dropout: float = 0.0,
        alpha: float = 1.0,
        beta: float = 1.0,
        time_scale: float = 200.0,
    ) -> None:
        super().__init__()
        self.alpha, self.beta, self.time_scale = alpha, beta, time_scale
        time_feats = hidden_channels * 4
        self.graph_conv = GraphConv(
            latent_dim + feature_dim,
            3,
            hidden_channels,
            edge_dim,
            time_feats,
            num_graph_layers,
            num_heads,
            dropout,
        )
        self.trans_conv = TransConv(
            hidden_channels,
            latent_dim,
            time_feats,
            model_channels,
            feature_dim,
            channel_mults,
            list(attention_levels),
            num_heads,
            dropout,
        )
        self.net_fc = nn.Linear(hidden_channels, 1)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: DenoiserCond
    ) -> torch.Tensor:
        """``z_t`` ``(B, N, latent)``, ``t`` ``(B,)``
        in ``[0, 1]``; returns ``(B, N, latent)``."""
        b, n, _ = z_t.shape
        real = (
            cond.key_pad_mask
            if cond.key_pad_mask is not None
            else torch.ones(b, n, dtype=torch.bool, device=z_t.device)
        )
        adj = cond.adjacency
        rows, i, j = torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=True)
        ci, cj = rows * n + i, rows * n + j
        net_ids = torch.arange(rows.numel(), device=z_t.device)
        pos = z_t[..., :2].reshape(b * n, 2)
        w = adj[rows, i, j].to(z_t.dtype)
        length = (pos[ci] - pos[cj]).abs().sum(-1)
        x_net = torch.stack([length, w, torch.full_like(w, 2.0)], dim=-1)
        x_cell = torch.cat([z_t, cond.features], dim=-1).reshape(b * n, -1)
        ei_c2n = torch.stack([torch.cat([ci, cj]), torch.cat([net_ids, net_ids])])
        ei_n2c = ei_c2n.flip(0)
        edge_attr = torch.stack(
            [
                torch.cat([w, w]),
                torch.ones(2 * w.numel(), device=z_t.device, dtype=z_t.dtype),
            ],
            dim=-1,
        )
        ts = t * self.time_scale
        h_cell, h_net = self.graph_conv(
            x_cell, x_net, ts.repeat_interleave(n), ts[rows], ei_c2n, ei_n2c, edge_attr
        )
        pred_cell = self.trans_conv(h_cell.view(b, n, -1), ts, cond.features, real)
        pred_net = self.net_fc(h_net)
        sign = torch.sign(pos[ci] - pos[cj]) * pred_net
        net_noise = (
            torch.zeros(b * n, 2, device=z_t.device, dtype=z_t.dtype)
            .index_add(0, ci, sign)
            .index_add(0, cj, -sign)
        )
        out = self.beta * pred_cell
        return torch.cat(
            [out[..., :2] + self.alpha * net_noise.view(b, n, 2), out[..., 2:]], dim=-1
        )
