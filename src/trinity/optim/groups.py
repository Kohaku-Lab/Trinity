"""Optimizer parameter groups for the backbone: plain AdamW or muP-scaled groups."""

import torch
from optimfactory import mup_param_group


def build_param_groups(
    params: list[torch.nn.Parameter],
    *,
    mode: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    base_dim: int = 256,
) -> list[dict]:
    """AdamW parameter groups for ``mode``.

    ``"adamw"`` returns one group; ``"mup"`` returns ``optimfactory.mup_param_group``'s
    per-layer learning-rate / weight-decay scaling relative to width ``base_dim``.
    """
    if mode == "adamw":
        return [{"params": params}]
    if mode == "mup":
        return mup_param_group(
            params,
            learning_rate,
            base_dim,
            weight_decay=weight_decay,
            weight_decay_scale=True,
        )
    raise ValueError(f"unknown optimizer mode {mode!r}; expected 'adamw' or 'mup'")
