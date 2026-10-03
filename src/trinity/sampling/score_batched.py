"""Batched torch version of the per-layout metric vector (``scoring.vector.COLS``).

``metric_vector_batched`` scores ``(B, N, 4)`` decoded boxes in the refiner's normalized
units (``scale = 1``, areas ``/ s^2``) against the per-row data of a ``RefineCase``: the
same columns, in the same order, as the numpy ``metric_vector``. Group reductions run by
scatter over the cluster / MIB ids; the bbox and the pairwise gaps are dense ``(B, N,
N)``.
"""

import torch

from trinity.floorplan.scoring.continuous import (
    W_BOUNDARY,
    W_COMPACT,
    W_FIXED,
    W_GROUP,
    W_MIB,
    W_OVERLAP,
)
from trinity.floorplan.scoring.cost import ALPHA, BETA, GAP_FLOOR
from trinity.floorplan.scoring.severity import (
    BETA_OVERLAP,
    TAU_BOUNDARY,
    TAU_GROUP,
    TAU_MIB,
)
from trinity.floorplan.scoring.vector import COLS
from trinity.losses import constraint as C
from trinity.sampling.refine_constraint import RefineCase

_EPS = 1e-9
_INF = float("inf")


def _pair_gap(g):
    """``(B, N, N)`` smallest axis translation to contact per pair.

    ``gx`` when the y ranges overlap, ``gy`` when
    the x ranges overlap, else ``min(gx, gy)``.
    """
    x, y, xr, yt = C._edges(g)
    gx = C._pair_gap(x, xr)
    gy = C._pair_gap(y, yt)
    y_overlap = (
        torch.minimum(yt[:, :, None], yt[:, None, :])
        - torch.maximum(y[:, :, None], y[:, None, :])
    ) > 0
    x_overlap = (
        torch.minimum(xr[:, :, None], xr[:, None, :])
        - torch.maximum(x[:, :, None], x[:, None, :])
    ) > 0
    return torch.where(y_overlap, gx, torch.where(x_overlap, gy, torch.minimum(gx, gy)))


def _group_stats(values, ids, member):
    """Per-id ``(sum, min, count)`` of ``values`` ``(B,
    N)`` over ``member`` rows, ids in ``[0, N]``."""
    b, n = ids.shape
    zeros = values.new_zeros(b, n + 1)
    total = zeros.scatter_add(
        1, ids, torch.where(member, values, torch.zeros_like(values))
    )
    member_values = torch.where(member, values, torch.full_like(values, _INF))
    lowest = torch.full_like(zeros, _INF).scatter_reduce(1, ids, member_values, "amin")
    count = zeros.scatter_add(1, ids, member.to(values.dtype))
    return total, lowest, count


def _drop_smallest_sum(slots, ids, member):
    """Sum over groups of ``> 1`` members of ``(sum of slots - smallest slot)``."""
    total, lowest, count = _group_stats(slots, ids, member)
    multi = count > 1
    kept = total - torch.where(multi, lowest, torch.zeros_like(lowest))
    return torch.where(multi, kept, torch.zeros_like(total)).sum(1)


def _nearest_same_gap(g, ids, mask):
    """``(B, N)`` gap to the nearest other member of the same group and the member mask.

    The gap is ``inf`` for a block alone in its group (then not a member).
    """
    member = (mask > 0.5) & (ids > 0)
    n = ids.shape[1]
    same = (
        (ids[:, :, None] == ids[:, None, :]) & member[:, :, None] & member[:, None, :]
    )
    same = same & ~torch.eye(n, dtype=torch.bool, device=g.device)[None]
    far = torch.full_like(g[..., 0, None], _INF).expand(-1, -1, n)
    gap = torch.where(same, _pair_gap(g), far).amin(2)
    return gap, member & torch.isfinite(gap)


def _boundary_terms(g, code, mask):
    """Per coded block: ``(sum of |required side -
    bbox edge|, required-side count, coded)``."""
    x, y, xr, yt = C._edges(g)
    x_min, y_min, x_max, y_max = C.bbox(g, mask)
    code = code.long()
    bits = [(code & 1 > 0), (code & 2 > 0), (code & 4 > 0), (code & 8 > 0)]
    dists = [
        (x - x_min[:, None]).abs(),
        (x_max[:, None] - xr).abs(),
        (y_max[:, None] - yt).abs(),
        (y - y_min[:, None]).abs(),
    ]
    coded = (mask > 0.5) & (code != 0)
    total = sum(b.to(g.dtype) * d for b, d in zip(bits, dists, strict=True))
    count = sum(b.to(g.dtype) for b in bits)
    total = torch.where(coded, total, torch.zeros_like(total))
    count = torch.where(coded, count, torch.zeros_like(count))
    return total, count, coded


def _masked_mean(values, include, denominator):
    """``sum(values[include]) / denominator`` per row, 0 where ``denominator == 0``."""
    summed = torch.where(include, values, torch.zeros_like(values)).sum(1)
    return torch.where(
        denominator > 0,
        summed / denominator.clamp_min(1),
        torch.zeros_like(denominator),
    )


def metric_vector_batched(g: torch.Tensor, case: RefineCase) -> torch.Tensor:
    """``(B, len(COLS))`` metric vectors of the decoded boxes ``g`` ``(B, N, 4)``."""
    mask = case.token_mask
    m = mask > 0.5
    mf = m.to(g.dtype)
    w, h = g[..., 2], g[..., 3]
    block_area = (w * h * mf).sum(1)
    s_n = (case.area_norm * mf).sum(1).clamp_min(0).sqrt()
    x_min, y_min, x_max, y_max = C.bbox(g, mask)
    bbox_n = (x_max - x_min) * (y_max - y_min)

    overlap = C.overlap_area(g, mask) / (block_area + _EPS)
    compact = bbox_n / (block_area + _EPS)
    hpwl_n, total_w = C.net_length(
        g,
        case.adjacency,
        case.pin_edge_xy_n,
        case.pin_edge_w,
        case.pin_edge_block,
        mask,
    )
    wl = hpwl_n / (total_w + _EPS)

    bd_total, bd_count, coded = _boundary_terms(g, case.boundary_code, mask)
    per_block = torch.where(
        coded, bd_total / bd_count.clamp_min(1), torch.zeros_like(bd_total)
    )
    n_coded = coded.to(g.dtype).sum(1)
    boundary = _masked_mean(per_block, coded, n_coded) / (s_n + _EPS)
    boundary = torch.where(n_coded > 0, boundary, torch.zeros_like(n_coded))

    gap, group_member = _nearest_same_gap(g, case.cluster_id, mask)
    gap0 = torch.where(group_member, gap, torch.zeros_like(gap))
    n_grouped = group_member.to(g.dtype).sum(1)
    group = _masked_mean(gap0, group_member, n_grouped) / (s_n + _EPS)
    group = torch.where(n_grouped > 0, group, torch.zeros_like(n_grouped))

    log_w = torch.log(w.clamp_min(_EPS))
    log_h = torch.log(h.clamp_min(_EPS))
    mib_member = m & (case.mib_id > 0)
    sum_w, _, mib_count = _group_stats(log_w, case.mib_id, mib_member)
    sum_h, _, _ = _group_stats(log_h, case.mib_id, mib_member)
    mean_w = (sum_w / mib_count.clamp_min(1)).gather(1, case.mib_id)
    mean_h = (sum_h / mib_count.clamp_min(1)).gather(1, case.mib_id)
    in_mib = mib_member & (mib_count.gather(1, case.mib_id) > 1)
    sq = (log_w - mean_w) ** 2 + (log_h - mean_h) ** 2
    sq = torch.where(in_mib, sq, torch.zeros_like(log_w))
    sq_sum, _, sq_count = _group_stats(sq, case.mib_id, in_mib)
    multi = sq_count > 1
    n_mib = multi.to(g.dtype).sum(1)
    group_var = torch.where(
        multi, sq_sum / sq_count.clamp_min(1), torch.zeros_like(sq_sum)
    )
    mib_var = torch.where(
        n_mib > 0, group_var.sum(1) / n_mib.clamp_min(1), torch.zeros_like(n_mib)
    )

    targets = case.target_n
    centers = g[..., :2] + g[..., 2:] / 2
    target_centers = targets[..., :2] + targets[..., 2:] / 2
    pos_known = (case.is_preplaced > 0.5) & (targets[..., 0] >= 0)
    shape_known = targets[..., 2] >= 0
    center_dist = torch.hypot(
        centers[..., 0] - target_centers[..., 0],
        centers[..., 1] - target_centers[..., 1],
    )
    d_pos = torch.where(
        pos_known, center_dist / (s_n[:, None] + _EPS), torch.zeros_like(w)
    )
    shape_err = (w - targets[..., 2]).abs() + (h - targets[..., 3]).abs()
    root_area = case.area_norm.clamp_min(0).sqrt()
    d_shape = torch.where(
        shape_known, shape_err / (root_area + _EPS), torch.zeros_like(w)
    )
    locked = m & ((case.is_fixed > 0.5) | (case.is_preplaced > 0.5))
    n_locked = locked.to(g.dtype).sum(1)
    fixed = _masked_mean(d_pos + d_shape, locked, n_locked)

    score_pre = (
        wl
        + W_COMPACT * compact
        + W_OVERLAP * overlap
        + W_BOUNDARY * boundary
        + W_GROUP * group
        + W_MIB * mib_var
        + W_FIXED * fixed
    )

    s = case.scale
    hpwl_gap = (s * hpwl_n - case.hpwl_base) / case.hpwl_base.clamp_min(GAP_FLOOR)
    area_gap = (s * s * bbox_n - case.area_base) / case.area_base.clamp_min(GAP_FLOOR)

    group_slots = torch.tanh(gap0 / (s_n[:, None] + _EPS) / TAU_GROUP)
    sev_group = _drop_smallest_sum(group_slots, case.cluster_id, group_member)
    deviation = (log_w - mean_w).abs() + (log_h - mean_h).abs()
    deviation = torch.where(in_mib, deviation, torch.zeros_like(log_w))
    sev_mib = _drop_smallest_sum(torch.tanh(deviation / TAU_MIB), case.mib_id, in_mib)
    boundary_slots = torch.tanh(bd_total / (s_n[:, None] + _EPS) / TAU_BOUNDARY)
    sev_boundary = torch.where(coded, boundary_slots, torch.zeros_like(bd_total)).sum(1)
    sev_rel = (sev_group + sev_mib + sev_boundary) / case.n_soft.clamp_min(1)
    quality = 1.0 + ALPHA * (hpwl_gap.clamp_min(0) + area_gap.clamp_min(0))
    soft_cost = quality * torch.exp(BETA * sev_rel) * torch.exp(BETA_OVERLAP * overlap)

    columns = [
        wl,
        compact,
        overlap,
        boundary,
        group,
        mib_var,
        fixed,
        score_pre,
        hpwl_gap,
        area_gap,
        sev_group,
        sev_mib,
        sev_boundary,
        sev_rel,
        soft_cost,
    ]
    out = torch.stack(columns, dim=-1)
    assert out.shape[-1] == len(COLS)
    return out
