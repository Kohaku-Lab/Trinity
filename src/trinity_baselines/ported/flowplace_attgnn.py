"""FlowPlace / ChipDiffusion ``AttGNN`` denoiser,
ported to the ``(z_t, t, cond) -> out`` interface.

Source: https://github.com/lamda-bbo/flowplace (licenses: NOTICE). Copied from
``flow_matching/networks/gnn_unet.py``, ``networks/mlp.py``, ``networks/vit.py``
(``AttentionBlock``, ``MultiHeadAttention``) and ``pos_encoding.py``
(``SinusoidContEncoding``), at the ``att-gnn.yaml`` + ``conv-layer/gat.yaml`` +
``size/large.yaml`` configuration. ChipDiffusion
(https://github.com/vint-1/chipdiffusion) uses the identical network.

Two changes from the original, both at the batch boundary:

* ``BatchWrapper`` assumed one graph shared by the whole batch. Here a batch is a padded
  ``(B, N)`` batch with a per-row dense adjacency, so :class:`RowGraphWrapper` builds
  each row's own ``edge_index`` from ``cond.adjacency`` (nonzero entries) offset by
  ``row * N``.
* Node features are the 19-D conditioning columns rather than
  their 2-D ``(w, h)``; the width is a constructor argument.

Registered as ``BASELINE_BACKBONE["flowplace_attgnn"]``; a config selects it by name.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.models import DenoiserCond
from trinity_baselines.registry import BASELINE_BACKBONE


class SinusoidContEncoding(nn.Module):
    """Sinusoidal embedding of a continuous ``(B,)`` time to ``(B, dim)``."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class MLP(nn.Module):
    """Their ``MLP``: ReLU stack, optional pre-LayerNorm and residual skip."""

    def __init__(
        self, num_layers, model_width, in_size, out_size, skip=False, layernorm=False
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    in_size if i == 0 else model_width,
                    out_size if i == num_layers - 1 else model_width,
                )
            )
            layers.append(nn.ReLU())
        self._nn = nn.Sequential(*layers[:-1])
        self._ln = nn.LayerNorm(in_size)
        self.skip = skip
        self.layernorm = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._nn(self._ln(x)) if self.layernorm else self._nn(x)
        return x + out if self.skip else out


class FiLM(nn.Module):
    """Their ``FiLM``: ``mult(cond) * x + add(cond)`` along the last axis."""

    def __init__(self, cond_dim: int, input_dim: int) -> None:
        super().__init__()
        self._mult_proj = nn.Linear(cond_dim, input_dim)
        self._add_proj = nn.Linear(cond_dim, input_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        mult = self._mult_proj(cond)[:, None, :]
        add = self._add_proj(cond)[:, None, :]
        return mult * x + add


class MultiHeadAttention(nn.Module):
    """Their unmasked multi-head self-attention over ``(B, T, C)``."""

    def __init__(self, num_heads, key_dim, value_dim, in_dim, out_dim) -> None:
        super().__init__()
        self.num_heads, self.key_dim, self.value_dim = num_heads, key_dim, value_dim
        self._k = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._q = nn.Linear(in_dim, num_heads * key_dim, bias=False)
        self._v = nn.Linear(in_dim, num_heads * value_dim, bias=False)
        self._out = nn.Linear(num_heads * value_dim, out_dim, bias=False)

    def forward(
        self, x: torch.Tensor, pad_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, _ = x.shape
        k = self._k(x).view(b, t, self.num_heads, self.key_dim).permute(0, 2, 3, 1)
        q = self._q(x).view(b, t, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
        v = self._v(x).view(b, t, self.num_heads, self.value_dim).permute(0, 2, 1, 3)
        logits = torch.matmul(q, k) / np.sqrt(self.key_dim)
        if pad_bias is not None:
            logits = logits + pad_bias
        out = (
            torch.matmul(F.softmax(logits, dim=-1), v)
            .permute(0, 2, 1, 3)
            .reshape(b, t, -1)
        )
        return self._out(out)


class AttentionBlock(nn.Module):
    """Their pre-LN attention block: attention residual then feed-forward residual."""

    def __init__(
        self, num_heads, model_dim, ff_num_layers, ff_size_factor, dropout
    ) -> None:
        super().__init__()
        self._attn = MultiHeadAttention(
            num_heads,
            model_dim // num_heads,
            model_dim // num_heads,
            model_dim,
            model_dim,
        )
        ff = []
        for i in range(ff_num_layers):
            ff.append(
                nn.Linear(
                    model_dim if i == 0 else ff_size_factor * model_dim,
                    model_dim if i == ff_num_layers - 1 else ff_size_factor * model_dim,
                )
            )
            if i < ff_num_layers - 1:
                ff.append(nn.ReLU())
        self._ff = nn.Sequential(*ff)
        self._ln1, self._ln2 = nn.LayerNorm(model_dim), nn.LayerNorm(model_dim)
        self._attn_dropout, self._ff_dropout = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, pad_bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self._attn_dropout(self._attn(self._ln1(x), pad_bias))
        return x + self._ff_dropout(self._ff(self._ln2(x)))


class RowGraphWrapper(nn.Module):
    """Run a PyG conv on a padded ``(B, N, F)`` batch whose rows are different graphs.

    ``adjacency`` is ``(B, N, N)``; every nonzero entry is an edge. Row ``b``'s node
    ``i`` becomes flat node ``b * N + i``. Edge attributes are the edge weights,
    expanded to ``edge_dim``.
    """

    def __init__(self, net: nn.Module, edge_dim: int) -> None:
        super().__init__()
        self.net = net
        self.edge_dim = edge_dim

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        b, n, f = x.shape
        rows, src, dst = torch.nonzero(adjacency, as_tuple=True)
        edge_index = torch.stack([rows * n + src, rows * n + dst])
        w = adjacency[rows, src, dst].unsqueeze(-1)
        edge_attr = w.expand(-1, self.edge_dim).to(x.dtype)
        out = self.net(x.reshape(b * n, f), edge_index, edge_attr=edge_attr)
        return out.view(b, n, -1)


def _gat(in_ch: int, out_ch: int, edge_dim: int, heads: int) -> RowGraphWrapper:
    """Their ``conv-layer/gat.yaml``: ``GATv2Conv`` with concatenated heads."""
    from torch_geometric.nn import GATv2Conv

    return RowGraphWrapper(
        GATv2Conv(in_ch, out_ch // heads, heads=heads, concat=True, edge_dim=edge_dim),
        edge_dim,
    )


class LinearEncoderLayer(nn.Module):
    """Their input projection with the optional
    sinusoidal encoding of the spatial input."""

    def __init__(self, in_features, out_features, input_encoding_dim=0) -> None:
        super().__init__()
        self._layer = nn.Linear(in_features, out_features)
        self.input_encoding_dim = input_encoding_dim
        self._encoding_layer = (
            nn.Linear(in_features * input_encoding_dim, out_features)
            if input_encoding_dim > 0
            else None
        )
        if input_encoding_dim > 0:
            half = input_encoding_dim // 2
            self.register_buffer(
                "freqs",
                torch.exp(
                    np.log(100.0) * torch.arange(half, dtype=torch.float32) / half
                ).view(1, 1, 1, half),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._layer(x)
        if self._encoding_layer is not None:
            theta = x.unsqueeze(-1) * self.freqs
            enc = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1).flatten(2)
            out = out + self._encoding_layer(enc)
        return out


class ResGNNBlock(nn.Module):
    """Their ``ResGNNBlock``: ``num_layers`` GAT+LN+Linear
    stages, FiLM time on the last, residual."""

    def __init__(
        self,
        hidden,
        hidden_node,
        cond_node,
        edge_dim,
        num_layers,
        t_dim,
        heads,
        dropout,
    ) -> None:
        super().__init__()
        self.hidden_node = hidden_node
        self._conv = nn.ModuleList(
            [
                _gat(
                    hidden + cond_node if i == 0 else hidden_node,
                    hidden_node,
                    edge_dim,
                    heads,
                )
                for i in range(num_layers)
            ]
        )
        self._lnorm = nn.ModuleList(
            [nn.LayerNorm(hidden_node) for _ in range(num_layers)]
        )
        self._linear = nn.ModuleList(
            [
                nn.Linear(hidden_node, hidden_node if i < num_layers - 1 else hidden)
                for i in range(num_layers)
            ]
        )
        self._cond = FiLM(t_dim, hidden_node)
        self._norm = nn.GroupNorm(1, hidden_node)
        self._act = nn.ReLU()
        self._drop = nn.Dropout(dropout)

    def forward(self, x, cond_x, adjacency, t_emb):
        skip = x
        x = torch.cat([x, cond_x], dim=-1)
        for conv, ln, lin in zip(
            self._conv[:-1], self._lnorm[:-1], self._linear[:-1], strict=True
        ):
            if x.shape[-1] == self.hidden_node:
                x = self._norm(x.movedim(-1, 1)).movedim(1, -1)
            x = self._drop(self._act(lin(ln(self._act(conv(x, adjacency))))))
        x = self._act(self._cond(self._conv[-1](x, adjacency), t_emb))
        return self._linear[-1](self._lnorm[-1](x)) + skip


class AttGNNBlock(nn.Module):
    """Their ``AttGNNBlock`` at ``num_layers=1``: one GAT,
    add embedded attention features, attention, out."""

    def __init__(
        self,
        hidden,
        att_node,
        cond_node,
        att_extra,
        edge_dim,
        t_dim,
        heads,
        dropout,
        num_heads,
        ff_num_layers,
        ff_size_factor,
    ) -> None:
        super().__init__()
        self._conv = _gat(hidden + cond_node, att_node, edge_dim, heads)
        self._att_embed = nn.Linear(cond_node + att_extra, att_node)
        self._attention = AttentionBlock(
            num_heads, att_node, ff_num_layers, ff_size_factor, dropout
        )
        self._lnorm = nn.LayerNorm(att_node)
        self._linear = nn.Linear(att_node, hidden)
        self._cond = FiLM(t_dim, att_node)
        self._act = nn.ReLU()

    def forward(self, x, cond_x, adjacency, t_emb, att_extra, pad_bias):
        skip = x
        x = self._act(
            self._cond(self._conv(torch.cat([x, cond_x], dim=-1), adjacency), t_emb)
        )
        x = x + self._att_embed(torch.cat([cond_x, att_extra], dim=-1))
        x = self._attention(x, pad_bias)
        return self._linear(self._lnorm(x)) + skip


@BASELINE_BACKBONE.register("flowplace_attgnn")
class FlowPlaceAttGNN(nn.Module):
    """The full ``AttGNN`` at the paper configuration, over the conditioning.

    ``latent_dim`` is the input/output width (2 for their ``(x, y)``; 3 for the 3-D
    latent) and ``feature_dim`` the per-node conditioning width (their
    ``cond_node_features``).
    """

    def __init__(
        self,
        latent_dim: int = 3,
        feature_dim: int = 19,
        hidden_size: int = 256,
        hidden_node_features: tuple[int, ...] = (256, 256, 256),
        attention_node_features: tuple[int, ...] = (256, 256, 256),
        layers_per_block: int = 2,
        t_encoding_dim: int = 32,
        input_encoding_dim: int = 32,
        edge_features: int = 4,
        gat_heads: int = 4,
        num_heads: int = 4,
        ff_num_layers: int = 2,
        ff_size_factor: int = 1,
        mlp_num_layers: int = 2,
        mlp_size_factor: int = 4,
        dropout: float = 0.0,
        dir_att_input: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.dir_att_input = dir_att_input
        self.t_encoder = SinusoidContEncoding(t_encoding_dim)
        self.encoder = LinearEncoderLayer(
            latent_dim + feature_dim, hidden_size, input_encoding_dim
        )
        blocks = []
        for hn, an in zip(hidden_node_features, attention_node_features, strict=True):
            blocks.append(
                ResGNNBlock(
                    hidden_size,
                    hn,
                    feature_dim,
                    edge_features,
                    layers_per_block,
                    t_encoding_dim,
                    gat_heads,
                    dropout,
                )
            )
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                blocks.append(
                    MLP(
                        mlp_num_layers,
                        mlp_size_factor * hidden_size,
                        hidden_size,
                        hidden_size,
                        skip=True,
                        layernorm=True,
                    )
                )
            blocks.append(
                AttGNNBlock(
                    hidden_size,
                    an,
                    feature_dim,
                    latent_dim,
                    edge_features,
                    t_encoding_dim,
                    gat_heads,
                    dropout,
                    num_heads,
                    ff_num_layers,
                    ff_size_factor,
                )
            )
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                blocks.append(
                    MLP(
                        mlp_num_layers,
                        mlp_size_factor * hidden_size,
                        hidden_size,
                        hidden_size,
                        skip=True,
                        layernorm=True,
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.decoder = nn.Linear(hidden_size, latent_dim)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: DenoiserCond
    ) -> torch.Tensor:
        """``z_t`` ``(B, N, latent)``, ``t`` ``(B,)``; returns ``(B, N, latent)``."""
        t_emb = self.t_encoder(t)
        cond_x = cond.features
        pad_bias = None
        if cond.key_pad_mask is not None:
            pad_bias = torch.where(cond.key_pad_mask, 0.0, -1e4)[:, None, None, :]
        x_skip = z_t
        x = self.encoder(torch.cat([z_t, cond_x], dim=-1))
        for block in self.blocks:
            if isinstance(block, AttGNNBlock):
                att_extra = x_skip if self.dir_att_input else x[..., : self.latent_dim]
                x = block(x, cond_x, cond.adjacency, t_emb, att_extra, pad_bias)
            elif isinstance(block, MLP):
                x = block(x)
            else:
                x = block(x, cond_x, cond.adjacency, t_emb)
        return self.decoder(x) + x_skip
