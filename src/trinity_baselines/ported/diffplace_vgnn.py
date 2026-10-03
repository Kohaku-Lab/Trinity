"""DiffPlace ``VectorGNNV2Global`` denoiser, ported to the ``(z_t, t, cond) -> out`` interface.

Source: https://github.com/HySonLab/DiffPlace (licenses: NOTICE). Copied from
``engine/networks/vector_gnn.py`` at the deployed configuration (``train.py`` /
``scripts/deploy.py``: hidden 256, 8 blocks x 2 vector message-passing layers, 8 heads, a global
supernode after every block; ``engine/diffplace.py``: 64-d time encoding, 4-d edge features).
Changes at the batch boundary:

* their message passing assumes one graph shared by the whole batch; here a batch is a padded
  ``(B, N)`` batch with a per-row dense adjacency, so nodes are flattened to ``b * N + i`` and
  each row's own edges (nonzero adjacency entries) are built per forward; the 4-d pin-offset
  edge attribute becomes the b2b weight repeated 4 times;
* node conditioning is the 19-D feature vector instead of their ``(w, h)``; the latent is 3-D
  ``(cx, cy, rho)``: the relative-position encoding uses the first two channels, the input
  concatenation, the output head and the skip use all three;
* the discrete rotation head is dropped (FloorSet blocks are not rotated); the global
  supernode's mean pooling is masked by ``key_pad_mask``; the integer-timestep embedding
  receives ``t * 1000`` (their 1000-step range).

Registered as ``BASELINE_BACKBONE["diffplace_vgnn"]``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trinity.models import DenoiserCond
from trinity_baselines.registry import BASELINE_BACKBONE


class SinusoidalPosEncoder(nn.Module):
    """Their 2-D sinusoidal position encoding ``(..., 2) -> (..., dim)``."""

    def __init__(self, dim: int, max_freq: float = 10.0) -> None:
        super().__init__()
        half = dim // 4
        emb = math.log(max_freq) / (half - 1) if half > 1 else 0.0
        self.register_buffer(
            "freqs", torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x = pos[..., 0:1] * self.freqs
        y = pos[..., 1:2] * self.freqs
        return torch.cat(
            [torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1
        )


class SinusoidalTimeEmbedding(nn.Module):
    """Their timestep embedding ``(B,) -> (B, dim)``."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        half = dim // 2
        self.register_buffer(
            "emb",
            torch.exp(
                torch.arange(half, dtype=torch.float32)
                * -(math.log(10000) / (half - 1))
            ),
        )
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.float()[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return F.pad(emb, (0, 1)) if self.dim % 2 else emb


class FiLM(nn.Module):
    """Their zero-initialized ``(1 + gamma) * x + beta`` with a per-node conditioning vector."""

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        for lin in (self.gamma, self.beta):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return (1 + self.gamma(cond)) * x + self.beta(cond)


class MLP(nn.Module):
    """Their MLP: Linear stack with LayerNorm + SiLU between layers."""

    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Linear(
                    in_dim if i == 0 else hidden_dim,
                    out_dim if i == num_layers - 1 else hidden_dim,
                )
            )
            if i < num_layers - 1:
                layers += [nn.LayerNorm(hidden_dim), nn.SiLU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _scatter_softmax(
    logits: torch.Tensor, index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Softmax of ``logits (E, H)`` over the edges sharing a destination node."""
    idx = index[:, None].expand_as(logits)
    mx = torch.full(
        (num_nodes, logits.shape[1]),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    mx = mx.scatter_reduce(0, idx, logits, reduce="amax", include_self=False).gather(
        0, idx
    )
    ex = torch.exp(logits - mx)
    den = torch.zeros(
        (num_nodes, logits.shape[1]), device=logits.device, dtype=logits.dtype
    )
    den = den.scatter_add(0, idx, ex).gather(0, idx)
    return ex / (den + 1e-8)


class VectorMessagePassingLayer(nn.Module):
    """Their vector message passing over flat nodes: relative-position-encoded messages,
    per-destination multi-head attention aggregation, residual update MLP + LayerNorm.
    """

    def __init__(
        self, hidden_dim, pos_encoding_dim, num_heads, dropout, edge_dim
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pos_encoder = SinusoidalPosEncoder(pos_encoding_dim)
        msg_in = 2 * hidden_dim + pos_encoding_dim + edge_dim
        self.message_mlp = MLP(msg_in, hidden_dim * 2, hidden_dim, 2, dropout)
        self.attn_proj = nn.Linear(msg_in, num_heads)
        self.update_mlp = MLP(hidden_dim * 2, hidden_dim * 2, hidden_dim, 2, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, edge_index, edge_attr):
        src, dst = edge_index
        m = x.shape[0]
        msg_in = torch.cat(
            [x[src], x[dst], self.pos_encoder(pos[src] - pos[dst]), edge_attr], dim=-1
        )
        messages = self.message_mlp(msg_in)
        attn = _scatter_softmax(self.attn_proj(msg_in), dst, m)
        weighted = (
            attn.unsqueeze(-1) * messages.view(-1, self.num_heads, self.head_dim)
        ).reshape(-1, messages.shape[-1])
        agg = torch.zeros_like(x).index_add(0, dst, weighted)
        return self.layer_norm(
            x + self.dropout(self.update_mlp(torch.cat([x, agg], dim=-1)))
        )


class VectorGNNBlock(nn.Module):
    """Their block: ``num_layers`` message-passing layers each followed by FiLM, then a FiLM-modulated feed-forward residual."""

    def __init__(
        self,
        hidden_dim,
        t_cond_dim,
        pos_encoding_dim,
        num_heads,
        num_layers,
        dropout,
        edge_dim,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VectorMessagePassingLayer(
                    hidden_dim, pos_encoding_dim, num_heads, dropout, edge_dim
                )
                for _ in range(num_layers)
            ]
        )
        self.films = nn.ModuleList(
            [FiLM(t_cond_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff_film = FiLM(t_cond_dim, hidden_dim)

    def forward(self, x, pos, edge_index, t_node, edge_attr):
        for layer, film in zip(self.layers, self.films, strict=True):
            x = film(layer(x, pos, edge_index, edge_attr), t_node)
        return self.ff_norm(x + self.ff_film(self.ff(x), t_node))


class InputProjection(nn.Module):
    """Their input projection: sinusoidal position encoding + raw latent + conditioning, plus the
    fixed-frequency spatial encoding of ``[latent, conditioning]``."""

    def __init__(
        self, latent_dim, cond_dim, hidden_dim, pos_encoding_dim, input_encoding_dim
    ) -> None:
        super().__init__()
        self.pos_encoder = SinusoidalPosEncoder(pos_encoding_dim)
        self.input_proj = nn.Linear(
            pos_encoding_dim + latent_dim + cond_dim, hidden_dim
        )
        self.input_encoding_dim = input_encoding_dim
        half = input_encoding_dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                math.log(100.0) * torch.arange(half, dtype=torch.float32) / half
            ).view(1, 1, 1, half),
        )
        self.encoding_proj = nn.Linear(
            (latent_dim + cond_dim) * input_encoding_dim, hidden_dim
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor, cond_x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(
            torch.cat([self.pos_encoder(z[..., :2]), z, cond_x], dim=-1)
        )
        spatial = torch.cat([z, cond_x], dim=-1)
        theta = spatial.unsqueeze(-1) * self.freqs
        enc = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1).flatten(2)
        return self.norm(h + self.encoding_proj(enc))


class OutputHead(nn.Module):
    """Their zero-initialized output head."""

    def __init__(self, hidden_dim, out_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_proj(F.silu(self.pre_proj(self.norm(x))))


class GlobalContextModule(nn.Module):
    """Their virtual global supernode: masked mean pool -> MLP -> gated broadcast add -> LayerNorm."""

    def __init__(self, hidden_dim, dropout=0.0) -> None:
        super().__init__()
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.process_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.process_mlp[-2].weight)
        nn.init.zeros_(self.process_mlp[-2].bias)

    def forward(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        w = real.to(h.dtype).unsqueeze(-1)
        pooled = (h * w).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1e-8)
        g = self.process_mlp(pooled + self.global_token).expand_as(h)
        return self.norm(h + self.gate(torch.cat([h, g], dim=-1)) * g)


@BASELINE_BACKBONE.register("diffplace_vgnn")
class DiffPlaceVGNN(nn.Module):
    """``VectorGNNV2Global`` at the deployed configuration, over the conditioning."""

    def __init__(
        self,
        latent_dim: int = 3,
        feature_dim: int = 19,
        hidden_size: int = 256,
        num_blocks: int = 8,
        layers_per_block: int = 2,
        num_heads: int = 8,
        t_encoding_dim: int = 64,
        pos_encoding_dim: int = 32,
        input_encoding_dim: int = 32,
        edge_features: int = 4,
        global_context_every: int = 1,
        dropout: float = 0.0,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        self.edge_features = edge_features
        self.time_scale = time_scale
        self.global_context_every = global_context_every
        self.input_proj = InputProjection(
            latent_dim, feature_dim, hidden_size, pos_encoding_dim, input_encoding_dim
        )
        self.t_encoder = SinusoidalTimeEmbedding(t_encoding_dim)
        self.time_cond_mlp = nn.Sequential(
            nn.Linear(t_encoding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [
                VectorGNNBlock(
                    hidden_size,
                    hidden_size,
                    pos_encoding_dim,
                    num_heads,
                    layers_per_block,
                    dropout,
                    edge_features,
                )
                for _ in range(num_blocks)
            ]
        )
        n_global = (num_blocks + global_context_every - 1) // global_context_every
        self.global_contexts = nn.ModuleList(
            [GlobalContextModule(hidden_size, dropout) for _ in range(n_global)]
        )
        self.position_head = OutputHead(hidden_size, latent_dim)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: DenoiserCond
    ) -> torch.Tensor:
        """``z_t`` ``(B, N, latent)``, ``t`` ``(B,)`` in ``[0, 1]``; returns ``(B, N, latent)``."""
        b, n, _ = z_t.shape
        real = (
            cond.key_pad_mask
            if cond.key_pad_mask is not None
            else torch.ones(b, n, dtype=torch.bool, device=z_t.device)
        )
        rows, src, dst = torch.nonzero(cond.adjacency, as_tuple=True)
        edge_index = torch.stack([rows * n + src, rows * n + dst])
        edge_attr = (
            cond.adjacency[rows, src, dst]
            .unsqueeze(-1)
            .expand(-1, self.edge_features)
            .to(z_t.dtype)
        )
        h = self.input_proj(z_t, cond.features).reshape(b * n, -1)
        pos = z_t[..., :2].reshape(b * n, 2)
        t_node = self.time_cond_mlp(
            self.t_encoder(t * self.time_scale)
        ).repeat_interleave(n, dim=0)
        g = 0
        for i, block in enumerate(self.blocks):
            h = block(h, pos, edge_index, t_node, edge_attr)
            if (i + 1) % self.global_context_every == 0:
                h = self.global_contexts[g](h.view(b, n, -1), real).reshape(b * n, -1)
                g += 1
        return self.position_head(h.view(b, n, -1)) + z_t
