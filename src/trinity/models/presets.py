"""The denoiser architecture config and named presets.

Component fields (``norm``, ``mlp``, ``attn``, ``time_cond``, ``graph_mix``) are
specs (a registry name, a dotted path, a ``{"name": ..., **kw}`` dict or a class).
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DenoiserArchConfig:
    """Everything the denoiser needs to build itself."""

    # (cx/s, cy/s, rho)
    latent_dim: int = 3
    feature_dim: int = 19
    # Graph PE width (0 = no PE); equals the GRAPH_PE builder's dim.
    graph_pe_dim: int = 0
    hidden: int = 384
    depth: int = 8
    heads: int = 8
    head_dim: int | None = None
    norm: Any = "layernorm"
    mlp: Any = "gelu"
    mlp_ratio: float = 4.0
    attn: Any = "sdpa_graph"
    qk_norm: bool = False
    graph_bias: bool = True
    # Message-passing step per block: none | mp |
    # {"name": "mp", "normalize": row|sym|none}.
    graph_mix: Any = "none"
    # adaln | adaln_shared | additive | token
    time_cond: Any = "adaln"
    # x0: the head emits the clean latent; target: the framing's own target (eps / v).
    output_kind: str = "x0"
    # Affine norm before the output head.
    post_norm: bool = False
    time_scale: float = 1000.0
    grad_ckpt: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden % self.heads != 0:
                raise ValueError(
                    f"hidden ({self.hidden}) not divisible by heads ({self.heads}); "
                    "set head_dim explicitly"
                )
            self.head_dim = self.hidden // self.heads


_SIZES: dict[str, dict[str, Any]] = {
    "S": dict(hidden=256, depth=6, heads=8),
    "B": dict(hidden=384, depth=8, heads=8),
    "L": dict(hidden=768, depth=14, heads=12),
    "XL": dict(hidden=1024, depth=20, heads=16),
}

PRESETS: dict[str, dict[str, Any]] = {}
for _size, _cfg in _SIZES.items():
    PRESETS[f"DiT-{_size}"] = dict(**_cfg)
    PRESETS[f"SiT-{_size}"] = dict(**_cfg)

PRESETS["ref-d384"] = dict(
    hidden=384, depth=8, heads=8, norm="layernorm", mlp="gelu", attn="naive_graph"
)
PRESETS["ref-d768"] = dict(
    hidden=768, depth=14, heads=12, norm="layernorm", mlp="gelu", attn="naive_graph"
)

PRESETS["modern-L"] = dict(
    hidden=768,
    depth=14,
    heads=12,
    norm="rmsnorm",
    mlp="swiglu",
    attn="sdpa_graph",
    qk_norm=True,
)


def get_preset(name: str, **overrides: Any) -> DenoiserArchConfig:
    """Build a :class:`DenoiserArchConfig` from a preset name plus field overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return DenoiserArchConfig(**{**PRESETS[name], **overrides})
