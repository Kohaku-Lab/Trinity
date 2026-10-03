"""Build the batched :class:`RefineCase` of a group of instances (normalized units).

One row per ``(case, draw)``; ``k`` consecutive rows share a case. The case carries what
the refiners and the batched scorer read: token and mobility masks, group ids, boundary
codes, the dense b2b matrix, the per-pin edges, the anchors, the fixed outline (``inf``
without one), the layout scale, the reference baselines ``hpwl_base`` / ``area_base``,
``N_soft`` and the normalized targets. Areas are ``/ s^2`` and lengths ``/ s``.
"""

import numpy as np
import torch

from trinity.conditioning.anchors import build_anchors
from trinity.conditioning.features import build_b2b_dense, pin_edges
from trinity.floorplan.scoring.cost import gap_baselines
from trinity.floorplan.scoring.soft import soft_denominator
from trinity.floorplan.types import Placement
from trinity.sampling.refine_constraint import RefineCase


def build_refine_case(group, k: int, device, dtype=torch.float32) -> RefineCase:
    """The ``RefineCase`` of ``len(group) * k`` rows for the instances in ``group``."""
    n = max(c.block_count for c in group)
    rows = len(group) * k
    n_pins = max((len(pin_edges(c)[1]) for c in group), default=1) or 1

    def floats(*shape):
        return torch.zeros(*shape, device=device, dtype=dtype)

    def longs(*shape):
        return torch.zeros(*shape, dtype=torch.long, device=device)

    def as_float(array):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=device)

    def as_long(array):
        return torch.as_tensor(np.asarray(array).astype(np.int64), device=device)

    area_norm, token_mask = floats(rows, n), floats(rows, n)
    mob_pos, mob_shape = floats(rows, n), floats(rows, n)
    cluster_id, mib_id, boundary_code = longs(rows, n), longs(rows, n), longs(rows, n)
    adjacency = floats(rows, n, n)
    pin_xy, pin_w, pin_block = (
        floats(rows, n_pins, 2),
        floats(rows, n_pins),
        longs(rows, n_pins),
    )
    anchor_z, anchor_mask = floats(rows, n, 3), floats(rows, n, 3)
    outline = torch.full((rows, 2), float("inf"), device=device, dtype=dtype)
    scale, hpwl_base, area_base, n_soft = (
        floats(rows),
        floats(rows),
        floats(rows),
        floats(rows),
    )
    target_n = torch.full((rows, n, 4), -1.0, device=device, dtype=dtype)
    is_fixed, is_preplaced = floats(rows, n), floats(rows, n)

    for gi, inst in enumerate(group):
        r, m = slice(gi * k, (gi + 1) * k), inst.block_count
        s = float(inst.s)
        area_norm[r, :m] = as_float(inst.area_targets / (s * s))
        token_mask[r, :m] = 1.0
        mob_pos[r, :m] = as_float(~inst.is_preplaced)
        mob_shape[r, :m] = as_float(~(inst.is_fixed | inst.is_preplaced))
        cluster_id[r, :m] = as_long(inst.cluster_id)
        mib_id[r, :m] = as_long(inst.mib_id)
        boundary_code[r, :m] = as_long(inst.boundary_code)
        adjacency[r, :m, :m] = as_float(build_b2b_dense(inst))
        edge_xy, edge_w, edge_block = pin_edges(inst)
        q = len(edge_w)
        if q:
            pin_xy[r, :q] = as_float(edge_xy / s)
            pin_w[r, :q] = as_float(edge_w)
            pin_block[r, :q] = torch.as_tensor(edge_block, device=device)
        case_anchor_z, case_anchor_mask, _ = build_anchors(inst, "group_mean")
        anchor_z[r, :m] = as_float(case_anchor_z)
        anchor_mask[r, :m] = as_float(case_anchor_mask)
        if inst.outline is not None:
            outline[r] = as_float(np.asarray(inst.outline, dtype=np.float64) / s)
        scale[r] = s
        empty = Placement(xywh=np.zeros((m, 4)), instance=inst)
        case_hpwl_base, case_area_base = gap_baselines(empty, fast_hpwl=True)
        hpwl_base[r], area_base[r] = case_hpwl_base, case_area_base
        n_soft[r] = soft_denominator(inst)
        targets = np.asarray(inst.target_positions, dtype=np.float64)
        target_n[r, :m] = as_float(np.where(targets >= 0, targets / s, -1.0))
        is_fixed[r, :m] = as_float(inst.is_fixed)
        is_preplaced[r, :m] = as_float(inst.is_preplaced)

    return RefineCase(
        area_norm=area_norm,
        token_mask=token_mask,
        mob_pos=mob_pos,
        mob_shape=mob_shape,
        cluster_id=cluster_id,
        mib_id=mib_id,
        boundary_code=boundary_code,
        adjacency=adjacency,
        pin_edge_xy_n=pin_xy,
        pin_edge_w=pin_w,
        pin_edge_block=pin_block,
        anchor_z=anchor_z,
        anchor_mask=anchor_mask,
        outline=outline,
        scale=scale,
        hpwl_base=hpwl_base,
        area_base=area_base,
        n_soft=n_soft,
        target_n=target_n,
        is_fixed=is_fixed,
        is_preplaced=is_preplaced,
    )
