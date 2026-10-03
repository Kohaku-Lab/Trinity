"""The loss-term interface.

The trainer holds a flat list of terms and calls each once per micro-batch with a shared
:class:`LossContext`. The regression terms read ``pred`` / ``target`` / ``weight``; the
aux terms read the decoded geometry (``xywh``) and the per-block data. Each term returns
``(loss_value, {log_name: value})``.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class LossContext:
    """Tensors a loss term may read for one training micro-batch.

    The regression fields (``pred``, ``target``, ``weight``, ``t``, ``anchor_mask``) are
    always set. The geometry fields are populated only when some configured term
    declares ``needs_geometry``; ``xywh`` and ``xywh_gt`` are the normalized ``(B, N,
    4)`` decodes of the predicted and ground-truth ``x0``, in the space where total
    block area is 1.
    """

    pred: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor
    t: torch.Tensor
    anchor_mask: torch.Tensor | None = None
    xywh: torch.Tensor | None = None
    xywh_gt: torch.Tensor | None = None
    area_targets: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    cluster_id: torch.Tensor | None = None
    boundary_code: torch.Tensor | None = None
    mib_id: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    pin_xy: torch.Tensor | None = None
    # (B, N, 2) per-block pin targets in / s units.
    pin_xy_n: torch.Tensor | None = None
    pin_w: torch.Tensor | None = None
    # (B, P, 2) per-pin positions in / s units,
    # (B, P) weights (0 = padding), (B, P) blocks.
    pin_edge_xy_n: torch.Tensor | None = None
    pin_edge_w: torch.Tensor | None = None
    pin_edge_block: torch.Tensor | None = None
    # (B, N) 1 = center movable (0 for preplaced).
    mob_pos: torch.Tensor | None = None
    # (B, N) 1 = shape movable (0 for fixed or preplaced).
    mob_shape: torch.Tensor | None = None
    # (B, N, N) dense b2b net weights.
    adjacency: torch.Tensor | None = None
    # (B,) reference bbox area / s^2, reference HPWL / s and N_soft per case.
    area_base: torch.Tensor | None = None
    hpwl_base: torch.Tensor | None = None
    n_soft: torch.Tensor | None = None
    extras: dict = field(default_factory=dict)


class LossTerm:
    """Base loss term; subclasses implement :meth:`__call__`.

    ``needs_geometry`` declares whether the term
    reads the decoded layout from the context.
    """

    name: str = "loss"
    needs_geometry: bool = False

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError
