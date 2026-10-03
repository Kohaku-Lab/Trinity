"""The ``ref_*`` auxiliary losses of the FloorSet reference recipe.

The terms read the normalized geometry the trainer puts in ``ctx.xywh`` (total block
area 1, so ``w * h == area / s^2``) and ``ctx.xywh_gt`` (the ground-truth decode).
Grouping, MIB and boundary reduce globally (one ``sum / sum`` over the batch); area and
overlap average over the batch. Each term's math is a pure ``fn`` so
``compile_loss_terms`` can compile it.

Variants (``variant=``):

* ``ref_area`` -- ``relu(bbox(pred) / bbox(gt) - 1)``; ``"soft"``
  uses a log-sum-exp bbox, ``"reference"`` the hard bbox.
* ``ref_overlap`` -- pairwise intersection area; ``"soft"`` from softplus penetrations,
  ``"reference"`` the raw ``ox * oy``.
* ``ref_grouping`` -- same-cluster pair gap-to-contact squared (``"gap"``) or center
  distance squared (``"reference"``).
* ``ref_mib`` -- same-MIB log-shape disagreement (``"log"``) or raw ``(dw)^2 + (dh)^2``
  (``"reference"``).
* ``ref_boundary`` -- squared distance of the coded sides to the bbox edges.
* ``ref_pin_hpwl`` -- ``sum pin_w * |center - pin|``.
"""

import torch

from trinity.losses.base import LossContext, LossTerm
from trinity.registry import LOSS

_BIG = 1e9


def _edges(xywh):
    """``(x_left, y_bottom, x_right, y_top)`` of every block."""
    x, y = xywh[..., 0], xywh[..., 1]
    return x, y, x + xywh[..., 2], y + xywh[..., 3]


def _abut_gap(x, y, xr, yt):
    """``(B, N, N)`` gap to contact ``max(gap_x,
    gap_y)`` (0 iff touching or overlapping)."""
    gx = (
        torch.maximum(x[:, :, None], x[:, None, :])
        - torch.minimum(xr[:, :, None], xr[:, None, :])
    ).clamp_min(0)
    gy = (
        torch.maximum(y[:, :, None], y[:, None, :])
        - torch.minimum(yt[:, :, None], yt[:, None, :])
    ).clamp_min(0)
    return torch.maximum(gx, gy)


def _bbox_edges(xywh: torch.Tensor, mask: torch.Tensor):
    """Per-sample bbox ``(x_min, y_min, x_max,
    y_max)`` over real tokens, four ``(B,)``."""
    x, y, xr, yt = _edges(xywh)
    m = mask > 0.5
    return (
        torch.where(m, x, torch.full_like(x, _BIG)).amin(1),
        torch.where(m, y, torch.full_like(y, _BIG)).amin(1),
        torch.where(m, xr, torch.full_like(xr, -_BIG)).amax(1),
        torch.where(m, yt, torch.full_like(yt, -_BIG)).amax(1),
    )


def _bbox_area(xywh: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sample bbox area over real tokens, ``(B,)``."""
    x_min, y_min, x_max, y_max = _bbox_edges(xywh, mask)
    return (x_max - x_min) * (y_max - y_min)


def _same_group_pairs(gid: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``(B, N, N)`` 1.0 for distinct real-token pairs in the same positive group."""
    same = (gid[:, :, None] == gid[:, None, :]) & (gid[:, :, None] > 0)
    eye = torch.eye(gid.shape[1], device=gid.device, dtype=torch.bool)[None]
    m = mask > 0.5
    return (same & ~eye & m[:, :, None] & m[:, None, :]).to(torch.get_default_dtype())


def _soft_bbox(xywh: torch.Tensor, mask: torch.Tensor, tau: float):
    """Per-sample bbox edges by log-sum-exp soft min / max with temperature ``tau``."""
    x, y, xr, yt = _edges(xywh)
    m = mask > 0.5
    xi = torch.where(m, x, torch.full_like(x, _BIG))
    yi = torch.where(m, y, torch.full_like(y, _BIG))
    xa = torch.where(m, xr, torch.full_like(xr, -_BIG))
    ya = torch.where(m, yt, torch.full_like(yt, -_BIG))
    x_min = -tau * torch.logsumexp(-xi / tau, dim=1)
    y_min = -tau * torch.logsumexp(-yi / tau, dim=1)
    x_max = tau * torch.logsumexp(xa / tau, dim=1)
    y_max = tau * torch.logsumexp(ya / tau, dim=1)
    return x_min, y_min, x_max, y_max


def _area(xywh, xywh_gt, token_mask, tau: float = 0.02):
    """Batch mean of ``relu(soft_bbox_area(pred) / bbox_area(gt) - 1)``."""
    x_min, y_min, x_max, y_max = _soft_bbox(xywh, token_mask, tau)
    pred_area = (x_max - x_min) * (y_max - y_min)
    gt_area = _bbox_area(xywh_gt, token_mask).clamp_min(1e-6)
    return torch.relu(pred_area / gt_area - 1.0).mean()


def _area_reference(xywh, xywh_gt, token_mask):
    """Batch mean of ``relu(bbox_area(pred) / bbox_area(gt) - 1)``."""
    pred_area = _bbox_area(xywh, token_mask)
    gt_area = _bbox_area(xywh_gt, token_mask).clamp_min(1e-6)
    return torch.relu(pred_area / gt_area - 1.0).mean()


def _pair_penetration(xywh: torch.Tensor, token_mask: torch.Tensor):
    """Per-pair axis penetrations ``(ox, oy)`` and
    the real-pair mask, each ``(B, N, N)``."""
    x, y, xr, yt = _edges(xywh)
    m = (token_mask > 0.5).to(xywh.dtype)
    ox = (
        torch.minimum(xr[:, :, None], xr[:, None, :])
        - torch.maximum(x[:, :, None], x[:, None, :])
    ).clamp_min(0)
    oy = (
        torch.minimum(yt[:, :, None], yt[:, None, :])
        - torch.maximum(y[:, :, None], y[:, None, :])
    ).clamp_min(0)
    return ox, oy, m[:, :, None] * m[:, None, :]


def _overlap(xywh, token_mask, tau: float = 0.01):
    """Batch mean of the pairwise ``softplus(ox) *
    softplus(oy)`` (softplus shifted to 0 at 0)."""
    ox, oy, pair = _pair_penetration(xywh, token_mask)
    zero = torch.nn.functional.softplus(torch.zeros_like(ox))
    sx = tau * torch.nn.functional.softplus(ox / tau) - tau * zero
    sy = tau * torch.nn.functional.softplus(oy / tau) - tau * zero
    return (torch.triu(sx * sy * pair, diagonal=1)).sum((1, 2)).mean()


def _overlap_reference(xywh, token_mask):
    """Batch mean of the total pairwise intersection area ``ox * oy``."""
    ox, oy, pair = _pair_penetration(xywh, token_mask)
    w, h = xywh[..., 2], xywh[..., 3]
    self_area = (w * h * (token_mask > 0.5).to(w.dtype)).sum(1)
    return (((ox * oy * pair).sum((1, 2)) - self_area).clamp_min(0.0) / 2.0).mean()


def _grouping(xywh, cluster_id, token_mask):
    """Mean over same-cluster pairs of the squared gap to contact."""
    cm = _same_group_pairs(cluster_id, token_mask)
    return (cm * _abut_gap(*_edges(xywh)).pow(2)).sum() / cm.sum().clamp_min(1.0)


def _grouping_reference(xywh, cluster_id, token_mask):
    """Mean over same-cluster pairs of the squared center distance."""
    cx = xywh[..., 0] + xywh[..., 2] / 2
    cy = xywh[..., 1] + xywh[..., 3] / 2
    cm = _same_group_pairs(cluster_id, token_mask)
    d2 = (cx[:, :, None] - cx[:, None, :]) ** 2 + (cy[:, :, None] - cy[:, None, :]) ** 2
    return (cm * d2).sum() / cm.sum().clamp_min(1.0)


def _mib(xywh, mib_id, token_mask):
    """Mean over same-MIB pairs of ``(dlog w)^2 + (dlog h)^2``."""
    lw = torch.log(xywh[..., 2].clamp_min(1e-6))
    lh = torch.log(xywh[..., 3].clamp_min(1e-6))
    mm = _same_group_pairs(mib_id, token_mask)
    ds = (lw[:, :, None] - lw[:, None, :]) ** 2 + (lh[:, :, None] - lh[:, None, :]) ** 2
    return (mm * ds).sum() / mm.sum().clamp_min(1.0)


def _mib_reference(xywh, mib_id, token_mask):
    """Mean over same-MIB pairs of ``(dw)^2 + (dh)^2``."""
    w, h = xywh[..., 2], xywh[..., 3]
    mm = _same_group_pairs(mib_id, token_mask)
    ds = (w[:, :, None] - w[:, None, :]) ** 2 + (h[:, :, None] - h[:, None, :]) ** 2
    return (mm * ds).sum() / mm.sum().clamp_min(1.0)


def _boundary(
    xywh: torch.Tensor, boundary_code: torch.Tensor, token_mask: torch.Tensor
):
    """Mean over the coded sides of the squared
    distance to the bbox edge (1=L 2=R 4=T 8=B)."""
    x, y, xr, yt = _edges(xywh)
    m = (token_mask > 0.5).to(x.dtype)
    x_min, y_min, x_max, y_max = (v.unsqueeze(1) for v in _bbox_edges(xywh, token_mask))
    code = boundary_code.long()
    left = (code & 1 > 0).to(x.dtype) * m
    right = (code & 2 > 0).to(x.dtype) * m
    top = (code & 4 > 0).to(x.dtype) * m
    bottom = (code & 8 > 0).to(x.dtype) * m
    distance = (
        left * (x - x_min) ** 2
        + right * (xr - x_max) ** 2
        + top * (yt - y_max) ** 2
        + bottom * (y - y_min) ** 2
    ).sum()
    sides = (left + right + top + bottom).sum()
    return distance / sides.clamp_min(1.0)


def _pin_hpwl(
    xywh: torch.Tensor, pin_xy: torch.Tensor, pin_w: torch.Tensor, token_mask
):
    """Batch mean of ``sum_b pin_w_b * |center_b - pin_b|_1``."""
    centers = xywh[..., :2] + xywh[..., 2:] / 2
    dist = (centers - pin_xy).abs().sum(-1)
    m = (token_mask > 0.5).to(centers.dtype)
    return (pin_w * dist * m).sum(1).mean()


class _AuxTerm(LossTerm):
    """An aux term: ``weight * fn(*_args(ctx))``;
    subclasses pick ``fn`` and ``_args``."""

    needs_geometry = True
    fn = None

    def __init__(self, weight: float) -> None:
        self.weight = weight

    def _args(self, ctx: LossContext) -> tuple:
        raise NotImplementedError

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict]:
        main = self.fn(*self._args(ctx))
        return self.weight * main, {self.name: main.detach()}


def _pick(variant: str, default_fn, reference_fn):
    """``reference_fn`` when ``variant == "reference"``, else ``default_fn``."""
    return reference_fn if variant == "reference" else default_fn


@LOSS.register("ref_area")
class AreaLoss(_AuxTerm):
    """Compactness against the ground truth;
    ``variant`` ``"soft"`` | ``"reference"``."""

    name = "area"

    def __init__(self, weight: float = 0.08, variant: str = "soft") -> None:
        super().__init__(weight)
        self.fn = _pick(variant, _area, _area_reference)

    def _args(self, ctx):
        return ctx.xywh, ctx.xywh_gt, ctx.token_mask


@LOSS.register("ref_overlap")
class OverlapLoss(_AuxTerm):
    """Pairwise overlap; ``variant`` ``"soft"`` | ``"reference"``."""

    name = "overlap"

    def __init__(self, weight: float = 0.15, variant: str = "soft") -> None:
        super().__init__(weight)
        self.fn = _pick(variant, _overlap, _overlap_reference)

    def _args(self, ctx):
        return ctx.xywh, ctx.token_mask


@LOSS.register("ref_grouping")
class GroupingLoss(_AuxTerm):
    """Same-cluster spread; ``variant`` ``"gap"`` | ``"reference"``."""

    name = "grouping"

    def __init__(self, weight: float = 0.2, variant: str = "gap") -> None:
        super().__init__(weight)
        self.fn = _pick(variant, _grouping, _grouping_reference)

    def _args(self, ctx):
        return ctx.xywh, ctx.cluster_id, ctx.token_mask


@LOSS.register("ref_mib")
class MIBLoss(_AuxTerm):
    """Same-MIB shape spread; ``variant`` ``"log"`` | ``"reference"``."""

    name = "mib"

    def __init__(self, weight: float = 0.2, variant: str = "log") -> None:
        super().__init__(weight)
        self.fn = _pick(variant, _mib, _mib_reference)

    def _args(self, ctx):
        return ctx.xywh, ctx.mib_id, ctx.token_mask


@LOSS.register("ref_boundary")
class BoundaryLoss(_AuxTerm):
    """Coded sides to the bbox edge, squared."""

    name = "boundary"
    fn = staticmethod(_boundary)

    def __init__(self, weight: float = 0.2) -> None:
        super().__init__(weight)

    def _args(self, ctx):
        return ctx.xywh, ctx.boundary_code, ctx.token_mask


@LOSS.register("ref_pin_hpwl")
class PinHPWLLoss(_AuxTerm):
    """Pin-to-block wirelength on the ``/ s`` pin targets ``pin_xy_n``."""

    name = "pin_hpwl"
    fn = staticmethod(_pin_hpwl)

    def __init__(self, weight: float = 0.0) -> None:
        super().__init__(weight)

    def _args(self, ctx):
        return ctx.xywh, ctx.pin_xy_n, ctx.pin_w, ctx.token_mask
