"""The denoiser backbone: a ``(z_t, t, cond) -> x0`` network over block tokens.

The network emits the clean latent ``x0`` (per block ``(cx/s, cy/s, rho)``), or the
framing's own target with ``output_kind="target"`` (:class:`X0View` converts it). The
per-block features (and graph PE) are concatenated to ``z_t`` at the input projection,
and the netlist adjacency is an additive attention bias inside each block.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from trinity.models.block import DenoiserBlock
from trinity.models.components.embed import InputEmbed, TimestepEmbedder
from trinity.models.graph_bias import GraphBias
from trinity.models.presets import DenoiserArchConfig
from trinity.registry import NORM, TIME_COND, build
from trinity.utils import modulate


@dataclass
class DenoiserCond:
    """The conditioning of a batch.

    The backbone reads ``features``, ``adjacency``, ``key_pad_mask`` and ``graph_pe``;
    the ``anchor_*`` and ``mib_soft_group`` fields are read by the sampling projections.
    """

    # (B, N, FEATURE_DIM) per-block features.
    features: torch.Tensor
    # (B, N, N) log1p b2b weights (None = no graph bias).
    adjacency: torch.Tensor | None = None
    # (B, N, latent) anchor values and (B, N, latent) 1 where an anchor is known.
    anchor_z: torch.Tensor | None = None
    anchor_mask: torch.Tensor | None = None
    # (B, N) bool, True = real token.
    key_pad_mask: torch.Tensor | None = None
    # (B, N, pe) graph PE.
    graph_pe: torch.Tensor | None = None
    # (B, N) all-soft MIB group id (0 = not a member).
    mib_soft_group: torch.Tensor | None = None


class FinalLayer(nn.Module):
    """Output head: norm + linear to the latent dim.

    With ``modulated`` it is the DiT adaLN-Zero head (norm modulated
    by ``c``); otherwise a plain non-affine norm and ``c`` is ignored.
    """

    def __init__(self, dim: int, latent_dim: int, modulated: bool = True) -> None:
        super().__init__()
        self.modulated = modulated
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaln = (
            nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim)) if modulated else None
        )
        self.linear = nn.Linear(dim, latent_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        if self.modulated:
            shift, scale = self.adaln(c).chunk(2, dim=-1)
            return self.linear(modulate(self.norm(x), shift, scale))
        return self.linear(self.norm(x))


class X0View(nn.Module):
    """A target-emitting backbone behind the ``(z_t, t, cond) -> x0`` interface.

    Converts the output with ``framing.x0_from_target``; holds no parameters of its own.
    """

    def __init__(self, backbone: nn.Module, framing) -> None:
        super().__init__()
        self.backbone = backbone
        self.framing = framing

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: DenoiserCond
    ) -> torch.Tensor:
        """Return the ``x0`` implied by the backbone's target-space output."""
        return self.framing.x0_from_target(z_t, self.backbone(z_t, t, cond), t)


class SetTransformerDenoiser(nn.Module):
    """Permutation-equivariant block-token denoiser (a DiT-style transformer)."""

    def __init__(self, config: DenoiserArchConfig) -> None:
        super().__init__()
        self.config = config
        self.grad_ckpt = config.grad_ckpt
        self.input_embed = InputEmbed(
            config.latent_dim, config.feature_dim, config.hidden, config.graph_pe_dim
        )
        self.t_embedder = TimestepEmbedder(config.hidden, time_scale=config.time_scale)
        self.graph_bias = GraphBias(enabled=config.graph_bias)
        self.time_cond = build(
            config.time_cond, TIME_COND, dim=config.hidden, depth=config.depth
        )
        modulated = self.time_cond.owns_block_modulation
        # The "token" strategy prepends a time token to the sequence.
        self.time_cond_prepends_token = type(self.time_cond).__name__ == "TokenCond"
        self.blocks = nn.ModuleList(
            DenoiserBlock(
                config.hidden,
                config.heads,
                config.head_dim,
                norm=config.norm,
                mlp=config.mlp,
                attn=config.attn,
                mlp_ratio=config.mlp_ratio,
                qk_norm=config.qk_norm,
                modulated=modulated,
                graph_mix=config.graph_mix,
            )
            for _ in range(config.depth)
        )
        self.post_norm = (
            build(config.norm, NORM, dim=config.hidden, affine=True)
            if config.post_norm
            else nn.Identity()
        )
        self.final = FinalLayer(config.hidden, config.latent_dim, modulated=modulated)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """DiT initialization: xavier linears, zeroed
        time-conditioning projections and head."""

        def _basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for module in self.time_cond.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        if self.final.adaln is not None:
            nn.init.zeros_(self.final.adaln[-1].weight)
            nn.init.zeros_(self.final.adaln[-1].bias)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    def _run_block(self, block, x, mod, bias, adj):
        if self.grad_ckpt and self.training:
            return checkpoint.checkpoint(block, x, mod, bias, adj, use_reentrant=False)
        return block(x, mod, bias, adj)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, cond: DenoiserCond
    ) -> torch.Tensor:
        """``z_t``: ``(B, N, latent_dim)``; ``t``:
        ``(B,)``; returns ``x0`` of the same shape."""
        x = self.input_embed(z_t, cond.features, cond.graph_pe)
        c = self.t_embedder(t)
        bias = self.graph_bias(cond.adjacency, cond.key_pad_mask)
        x, bias = self.time_cond.enter(x, c, bias, key_pad_mask=cond.key_pad_mask)
        # One modulation per block (None for the unmodulated strategies).
        mods = self.time_cond.modulation(c)
        adj = cond.adjacency
        if adj is not None and self.time_cond_prepends_token:
            adj = nn.functional.pad(adj, (1, 0, 1, 0))
        for block, mod in zip(self.blocks, mods, strict=True):
            x = self._run_block(block, x, mod, bias, adj)
        x = self.time_cond.exit(x)
        x = self.post_norm(x)
        return self.final(x, c)
