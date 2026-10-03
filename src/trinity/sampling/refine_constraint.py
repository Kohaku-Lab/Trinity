"""Constraint refiner by autograd: gradient descent
of the aux terms on a finished sample's latent.

The energy is the weighted sum of the constraint quantities of ``losses/constraint.py``
(``overlap``, ``group``, ``mib``, ``boundary``, ``wl``, ``area``, plus ``outline`` for a
case with a fixed outline), evaluated on the decoded geometry in normalized units
(``scale = 1``, areas ``/ s^2``, pin edges ``/ s``). Frozen channels are detached by the
terms' own masks (``mob_pos`` / ``mob_shape``); anchored coordinates are re-clamped
after every step.

``refine_latent`` is the batched entry point over one :class:`RefineCase`;
:class:`ConstraintRefiner` is the ``REFINER`` ``constraint_latent`` (``refine(z0,
case)``), the autograd reference of the closed-form refiner.
"""

import math
from dataclasses import dataclass

import torch

from trinity.decode import z_to_xywh
from trinity.losses import constraint as C
from trinity.registry import REFINER


def _outline_fn(g, c):
    """``(B,)`` bbox width and height beyond the
    fixed outline, ``/ s`` (0 without an outline)."""
    if c.outline is None:
        return g.new_zeros(g.shape[0])
    x_min, y_min, x_max, y_max = C.bbox(
        C._freeze(g, c.mob_pos, c.mob_shape), c.token_mask
    )
    width_excess = (x_max - x_min - c.outline[:, 0]).clamp_min(0)
    height_excess = (y_max - y_min - c.outline[:, 1]).clamp_min(0)
    _, s = C._scales(c.area_norm, c.token_mask)
    return (width_excess + height_excess) / s


TERMS = {
    "overlap": lambda g, c: C._overlap_fn(
        g, c.mob_pos, c.mob_shape, c.area_norm, c.token_mask
    ),
    "group": lambda g, c: C._group_fn(
        g, c.mob_pos, c.mob_shape, c.area_norm, c.token_mask, c.cluster_id
    ),
    "mib": lambda g, c: C._mib_fn(g, c.mob_pos, c.mob_shape, c.token_mask, c.mib_id),
    "boundary": lambda g, c: C._boundary_fn(
        g, c.mob_pos, c.mob_shape, c.area_norm, c.token_mask, c.boundary_code
    ),
    "wl": lambda g, c: C._wl_fn(
        g,
        c.mob_pos,
        c.mob_shape,
        c.area_norm,
        c.token_mask,
        c.adjacency,
        c.pin_edge_xy_n,
        c.pin_edge_w,
        c.pin_edge_block,
    ),
    "area": lambda g, c: C._area_fn(
        g, c.mob_pos, c.mob_shape, c.area_norm, c.token_mask
    ),
    "outline": _outline_fn,
}


@dataclass
class RefineCase:
    """The static per-row tensors of a refine
    batch (all ``(B, ...)``, normalized units).

    ``outline`` is the fixed outline ``(B, 2)`` of a bookshelf protocol (``None`` or
    ``inf`` without one); the fields from ``scale`` on are read by the batched scorer.
    """

    area_norm: torch.Tensor
    token_mask: torch.Tensor
    mob_pos: torch.Tensor
    mob_shape: torch.Tensor
    cluster_id: torch.Tensor
    mib_id: torch.Tensor
    boundary_code: torch.Tensor
    adjacency: torch.Tensor
    pin_edge_xy_n: torch.Tensor
    pin_edge_w: torch.Tensor
    pin_edge_block: torch.Tensor
    anchor_z: torch.Tensor | None = None
    anchor_mask: torch.Tensor | None = None
    outline: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    hpwl_base: torch.Tensor | None = None
    area_base: torch.Tensor | None = None
    n_soft: torch.Tensor | None = None
    target_n: torch.Tensor | None = None
    is_fixed: torch.Tensor | None = None
    is_preplaced: torch.Tensor | None = None


def constraint_energy(
    z: torch.Tensor, case: RefineCase, weights: dict[str, float]
) -> torch.Tensor:
    """Per-sample ``sum_term weight_term * term(decode(z))``, ``(B,)``."""
    ones = z.new_ones(z.shape[0])
    g = z_to_xywh(z, case.area_norm, ones)
    total = z.new_zeros(z.shape[0])
    for name, w in weights.items():
        if w:
            total = total + w * TERMS[name](g, case)
    return total


def lr_factor(schedule: str, step: int, steps: int) -> float:
    """The lr multiplier of ``schedule`` (constant
    / cosine / linear decay to 0) at ``step``."""
    t = step / max(steps - 1, 1)
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "linear":
        return 1.0 - t
    return 1.0


@torch.enable_grad()
def refine_latent(
    z0: torch.Tensor,
    case: RefineCase,
    weights: dict[str, float],
    steps: int,
    lr: float,
    optimizer: str = "adam",
    snapshots: tuple[int, ...] = (),
    lr_schedule: str = "constant",
    betas: tuple[float, float] = (0.9, 0.999),
) -> dict[int, torch.Tensor]:
    """Descend the constraint energy from ``z0`` ``(B, N, 3)`` with autograd.

    ``optimizer`` is ``adam`` or ``sgd``. Returns ``{step: z}`` (anchors
    clamped) at every step in ``snapshots`` (0 = the input) and at ``steps``.
    """
    z = z0.detach().clone().requires_grad_(True)
    if optimizer == "adam":
        opt = torch.optim.Adam([z], lr=lr, betas=betas)
    else:
        opt = torch.optim.SGD([z], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: lr_factor(lr_schedule, i, steps)
    )
    want = set(snapshots) | {steps}

    def clamp(t):
        if case.anchor_mask is None:
            return t
        return torch.where(case.anchor_mask > 0.5, case.anchor_z, t)

    out = {0: clamp(z).detach().clone()} if 0 in want else {}
    for i in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        constraint_energy(clamp(z), case, weights).sum().backward()
        opt.step()
        sched.step()
        if i in want:
            out[i] = clamp(z).detach().clone()
    return out


@REFINER.register("constraint_latent")
class ConstraintRefiner:
    """The autograd constraint refiner (``refine(z0, case) -> z``).

    ``weight`` applies to every term unless ``weights`` gives them one by one.
    """

    def __init__(
        self,
        steps: int = 100,
        lr: float = 0.01,
        weight: float = 1.0,
        weights: dict | None = None,
        optimizer: str = "adam",
        lr_schedule: str = "constant",
        betas: tuple[float, float] = (0.9, 0.999),
    ) -> None:
        self.steps = steps
        self.lr = lr
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.betas = tuple(betas)
        self.weights = (
            {name: weight for name in TERMS} if weights is None else dict(weights)
        )

    def refine(self, z0: torch.Tensor, case: RefineCase) -> torch.Tensor:
        """``z0`` ``(B, N, 3)`` after ``steps`` updates against ``case``."""
        snapshots = refine_latent(
            z0,
            case,
            self.weights,
            self.steps,
            self.lr,
            self.optimizer,
            lr_schedule=self.lr_schedule,
            betas=self.betas,
        )
        return snapshots[self.steps]
