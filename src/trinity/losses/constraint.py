"""Auxiliary loss terms: one per constraint, each the constraint's own violated quantity.

Design record: ``docs/aux-terms.md``. Six independent terms on the decoded geometry ``xywh``
``(B, N, 4)``; each has one tunable, ``weight``, optionally applied per sample as ``weight * t``
(``t_weight``). Every normalizer is a property of the instance: ``s² = Σ w h`` over real blocks
and ``s = sqrt(s²)``. Fixed / preplaced channels are detached in every term. The math of each
term is a pure function (``fn``) so ``compile_loss_terms`` can ``torch.compile`` it.

* ``overlap``  -- Σ pairwise intersection area / s²
* ``group``    -- Σ clusters (minimum-spanning-tree total gap-to-contact) / s
* ``mib``      -- Σ groups Σ members |log-aspect − group median log-aspect|
* ``boundary`` -- Σ coded blocks Σ required sides (side − bbox edge) / s
* ``wl``       -- weighted Manhattan net length (b2b + per-pin p2b) / (s · Σ weights)
* ``area``     -- bbox area / s²
"""

import torch

from trinity.losses.base import LossContext, LossTerm
from trinity.registry import LOSS

_EPS = 1e-9
# Upper bound on the size of a cluster / MIB group (iterations of the batched Prim).
MAX_GROUP_SIZE = 16


def _freeze(xywh, mob_pos, mob_shape):
    """``xywh`` with the centers of position-frozen and the sizes of shape-frozen blocks detached.

    The corner form is rebuilt from the (partly detached) centers and sizes.
    """
    mp = mob_pos.unsqueeze(-1) > 0.5
    ms = mob_shape.unsqueeze(-1) > 0.5
    wh = xywh[..., 2:]
    c = xywh[..., :2] + wh / 2
    c = torch.where(mp, c, c.detach())
    wh = torch.where(ms, wh, wh.detach())
    return torch.cat([c - wh / 2, wh], dim=-1)


def _real_pairs(mask):
    """``(B, N, N)`` 1.0 for distinct real-token pairs."""
    m = mask > 0.5
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=torch.bool)[None]
    return (m[:, :, None] & m[:, None, :] & ~eye).to(torch.get_default_dtype())


def _scales(area_targets, mask):
    """Per-sample ``(s², s)`` from the real blocks' areas."""
    m = (mask > 0.5).to(area_targets.dtype)
    s2 = (area_targets * m).sum(1).clamp_min(_EPS)
    return s2, s2.sqrt()


def _edges(g):
    """``(x_left, y_bottom, x_right, y_top)`` of every block."""
    x, y = g[..., 0], g[..., 1]
    return x, y, x + g[..., 2], y + g[..., 3]


def overlap_area(g, mask):
    """``(B,)`` sum over real pairs of the rectangle intersection area."""
    x, y, xr, yt = _edges(g)
    ox = _pair_overlap(x, xr)
    oy = _pair_overlap(y, yt)
    return torch.triu(ox * oy * _real_pairs(mask), diagonal=1).sum((1, 2))


def _pair_overlap(lo, hi):
    """``(B, N, N)`` overlap of every pair of intervals ``[lo, hi]`` (0 when disjoint)."""
    overlap = torch.minimum(hi[:, :, None], hi[:, None, :]) - torch.maximum(
        lo[:, :, None], lo[:, None, :]
    )
    return overlap.clamp_min(0)


def _pair_gap(lo, hi):
    """``(B, N, N)`` gap between every pair of intervals ``[lo, hi]`` (0 when they meet)."""
    gap = torch.maximum(lo[:, :, None], lo[:, None, :]) - torch.minimum(
        hi[:, :, None], hi[:, None, :]
    )
    return gap.clamp_min(0)


def gap_matrix(g):
    """``(B, N, N)`` distance to contact ``max(gap_x, gap_y)`` (0 iff touching or overlapping)."""
    x, y, xr, yt = _edges(g)
    return torch.maximum(_pair_gap(x, xr), _pair_gap(y, yt))


def cluster_mst_gap(g, cluster_id, mask):
    """``(B,)`` sum over clusters of the minimum-spanning-tree total gap, by batched Prim.

    Runs ``MAX_GROUP_SIZE - 1`` iterations; group ids are indexed up to ``N``.
    """
    gap = gap_matrix(g)
    b, n = cluster_id.shape
    member = (mask > 0.5) & (cluster_id > 0)
    same = (
        (cluster_id[:, :, None] == cluster_id[:, None, :])
        & member[:, :, None]
        & member[:, None, :]
    )
    same = same & ~torch.eye(n, dtype=torch.bool, device=g.device)[None]
    idx = torch.arange(n, device=g.device)[None].expand(b, n)
    inf = torch.full_like(gap, float("inf"))
    big = torch.full((b, n + 1), n, device=g.device, dtype=torch.long)
    first = big.scatter_reduce(1, cluster_id, torch.where(member, idx, n), "amin")
    in_tree = member & (idx == first.gather(1, cluster_id))
    total = torch.zeros(b, device=g.device, dtype=gap.dtype)
    for _ in range(MAX_GROUP_SIZE - 1):
        allowed = same & in_tree[:, None, :] & ~in_tree[:, :, None]
        cand = torch.where(allowed, gap, inf).amin(2)
        has = torch.isfinite(cand)
        cmin = torch.full((b, n + 1), float("inf"), device=g.device, dtype=gap.dtype)
        cmin = cmin.scatter_reduce(1, cluster_id, cand, "amin")
        is_min = has & (cand <= cmin.gather(1, cluster_id))
        first_min = big.scatter_reduce(
            1, cluster_id, torch.where(is_min, idx, n), "amin"
        )
        chosen = is_min & (idx == first_min.gather(1, cluster_id))
        total = total + torch.where(chosen, cand, torch.zeros_like(cand)).sum(1)
        in_tree = in_tree | chosen
    return total


def mib_median_deviation(g, mib_id, mask):
    """``(B,)`` sum over MIB members of ``|log-aspect - group median log-aspect|``.

    The group median is detached; group ids are indexed up to ``N``.
    """
    rho = torch.log(g[..., 2].clamp_min(_EPS)) - torch.log(g[..., 3].clamp_min(_EPS))
    b, n = mib_id.shape
    member = (mask > 0.5) & (mib_id > 0)
    gid = torch.arange(n + 1, device=g.device)[None, :, None]
    onehot = member[:, None, :] & (mib_id[:, None, :] == gid)
    expanded = rho[:, None, :].expand(b, n + 1, n)
    vals = torch.where(onehot, expanded, torch.full_like(expanded, float("inf")))
    ordered = vals.sort(dim=2).values
    cnt = onehot.sum(2)
    lo = ((cnt - 1) // 2).clamp_min(0)
    hi = (cnt // 2).clamp_min(0)
    med = 0.5 * (
        ordered.gather(2, lo[:, :, None]) + ordered.gather(2, hi[:, :, None])
    ).squeeze(2)
    med_block = med.gather(1, mib_id).detach()
    in_group = member & (cnt.gather(1, mib_id) > 1)
    dev = (rho - med_block).abs()
    return torch.where(in_group, dev, torch.zeros_like(dev)).sum(1)


def bbox(g, mask):
    """Per-sample bounding box ``(x_min, y_min, x_max, y_max)`` over real blocks."""
    x, y, xr, yt = _edges(g)
    m = mask > 0.5
    big = 1e9
    return (
        torch.where(m, x, torch.full_like(x, big)).amin(1),
        torch.where(m, y, torch.full_like(y, big)).amin(1),
        torch.where(m, xr, torch.full_like(xr, -big)).amax(1),
        torch.where(m, yt, torch.full_like(yt, -big)).amax(1),
    )


def boundary_distance(g, boundary_code, mask):
    """``(B,)`` sum over coded blocks of the distance from each required side to its bbox edge."""
    x, y, xr, yt = _edges(g)
    x_min, y_min, x_max, y_max = bbox(g, mask)
    code = boundary_code.long()
    m = (mask > 0.5).to(x.dtype)
    d = (
        (code & 1 > 0).to(x.dtype) * (x - x_min[:, None])
        + (code & 2 > 0).to(x.dtype) * (x_max[:, None] - xr)
        + (code & 4 > 0).to(x.dtype) * (y_max[:, None] - yt)
        + (code & 8 > 0).to(x.dtype) * (y - y_min[:, None])
    )
    return (d * m).sum(1)


def net_length(g, adjacency, pin_xy, pin_w, pin_block, mask):
    """Per sample, the weighted Manhattan net length (b2b + per-pin p2b) and the total weight."""
    c = g[..., :2] + g[..., 2:] / 2
    pair = _real_pairs(mask)
    d = (c[:, :, None, 0] - c[:, None, :, 0]).abs() + (
        c[:, :, None, 1] - c[:, None, :, 1]
    ).abs()
    b2b = (adjacency * d * pair).sum((1, 2)) * 0.5
    w_b2b = (adjacency * pair).sum((1, 2)) * 0.5
    cb = c.gather(1, pin_block[:, :, None].expand(-1, -1, 2))
    p2b = (pin_w * (cb - pin_xy).abs().sum(-1)).sum(1)
    return b2b + p2b, w_b2b + pin_w.sum(1)


def _overlap_fn(xywh, mob_pos, mob_shape, area_targets, mask):
    """``overlap`` term: pairwise intersection area / s^2."""
    s2, _ = _scales(area_targets, mask)
    return overlap_area(_freeze(xywh, mob_pos, mob_shape), mask) / s2


def _group_fn(xywh, mob_pos, mob_shape, area_targets, mask, cluster_id):
    """``group`` term: cluster MST gap / s."""
    _, s = _scales(area_targets, mask)
    return cluster_mst_gap(_freeze(xywh, mob_pos, mob_shape), cluster_id, mask) / s


def _mib_fn(xywh, mob_pos, mob_shape, mask, mib_id):
    """``mib`` term: deviation from the group median log-aspect."""
    return mib_median_deviation(_freeze(xywh, mob_pos, mob_shape), mib_id, mask)


def _boundary_fn(xywh, mob_pos, mob_shape, area_targets, mask, boundary_code):
    """``boundary`` term: distance of the coded sides to the bbox / s."""
    _, s = _scales(area_targets, mask)
    return boundary_distance(_freeze(xywh, mob_pos, mob_shape), boundary_code, mask) / s


def _wl_fn(
    xywh, mob_pos, mob_shape, area_targets, mask, adjacency, pin_xy, pin_w, pin_block
):
    """``wl`` term: net length / (s * total net weight)."""
    _, s = _scales(area_targets, mask)
    g = _freeze(xywh, mob_pos, mob_shape)
    length, weight = net_length(g, adjacency, pin_xy, pin_w, pin_block, mask)
    return length / (s * weight.clamp_min(_EPS))


def _area_fn(xywh, mob_pos, mob_shape, area_targets, mask):
    """``area`` term: bbox area / s^2."""
    s2, _ = _scales(area_targets, mask)
    x_min, y_min, x_max, y_max = bbox(_freeze(xywh, mob_pos, mob_shape), mask)
    return (x_max - x_min) * (y_max - y_min) / s2


class _ConstraintTerm(LossTerm):
    """The batch mean of ``fn`` per sample, weighted by ``weight`` (or ``weight * t``)."""

    needs_geometry = True
    fn = None

    def __init__(self, weight: float = 0.01, t_weight: bool = False) -> None:
        self.weight = weight
        self.t_weight = t_weight

    def _args(self, ctx: LossContext) -> tuple:
        raise NotImplementedError

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict]:
        v = self.fn(*self._args(ctx))
        w = self.weight * ctx.t if self.t_weight else self.weight
        return (w * v).mean(), {self.name: v.mean().detach()}


@LOSS.register("overlap")
class OverlapLoss(_ConstraintTerm):
    name = "overlap"
    fn = staticmethod(_overlap_fn)

    def _args(self, ctx):
        return ctx.xywh, ctx.mob_pos, ctx.mob_shape, ctx.area_targets, ctx.token_mask


@LOSS.register("group")
class GroupLoss(_ConstraintTerm):
    name = "group"
    fn = staticmethod(_group_fn)

    def _args(self, ctx):
        return (
            ctx.xywh,
            ctx.mob_pos,
            ctx.mob_shape,
            ctx.area_targets,
            ctx.token_mask,
            ctx.cluster_id,
        )


@LOSS.register("mib")
class MIBLoss(_ConstraintTerm):
    name = "mib"
    fn = staticmethod(_mib_fn)

    def _args(self, ctx):
        return ctx.xywh, ctx.mob_pos, ctx.mob_shape, ctx.token_mask, ctx.mib_id


@LOSS.register("boundary")
class BoundaryLoss(_ConstraintTerm):
    name = "boundary"
    fn = staticmethod(_boundary_fn)

    def _args(self, ctx):
        return (
            ctx.xywh,
            ctx.mob_pos,
            ctx.mob_shape,
            ctx.area_targets,
            ctx.token_mask,
            ctx.boundary_code,
        )


@LOSS.register("wl")
class WirelengthLoss(_ConstraintTerm):
    name = "wl"
    fn = staticmethod(_wl_fn)

    def _args(self, ctx):
        return (
            ctx.xywh,
            ctx.mob_pos,
            ctx.mob_shape,
            ctx.area_targets,
            ctx.token_mask,
            ctx.adjacency,
            ctx.pin_edge_xy_n,
            ctx.pin_edge_w,
            ctx.pin_edge_block,
        )


@LOSS.register("area")
class AreaLoss(_ConstraintTerm):
    name = "area"
    fn = staticmethod(_area_fn)

    def _args(self, ctx):
        return ctx.xywh, ctx.mob_pos, ctx.mob_shape, ctx.area_targets, ctx.token_mask
