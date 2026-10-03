"""Soft-constraint violation counts (grouping, MIB, boundary) and ``V_rel``.

* ``V_grouping = sum_p (c_p - 1)`` -- ``c_p`` connected components of cluster ``p``;
* ``V_mib = sum_q (s_q - 1)`` -- ``s_q`` distinct ``(round(w, 4), round(h, 4))`` shapes of
  MIB group ``q``;
* ``V_boundary`` -- coded blocks missing a required bbox edge (within ``1e-6``).

``V_rel = (V_grouping + V_boundary + V_mib) / N_soft`` with ``N_soft`` the largest possible
count (:func:`soft_denominator`), so ``V_rel`` lies in ``[0, 1]``.
"""

from dataclasses import dataclass, field

import numpy as np

from trinity.floorplan.geometry import bounding_box, connected_components
from trinity.floorplan.types import FloorplanInstance, Placement


@dataclass
class SoftReport:
    """Soft-constraint violations and the resulting ``V_rel``."""

    grouping: int = 0
    mib: int = 0
    boundary: int = 0
    boundary_blocks: list[int] = field(default_factory=list)
    grouping_groups: list[int] = field(default_factory=list)
    mib_groups: list[int] = field(default_factory=list)
    n_soft: int = 0

    @property
    def total(self) -> int:
        return self.grouping + self.mib + self.boundary

    @property
    def v_rel(self) -> float:
        return self.total / max(self.n_soft, 1)


def check_grouping(placement: Placement) -> tuple[int, list[int]]:
    """``V_grouping = sum_p (c_p - 1)``; returns ``(violations, split cluster ids)``."""
    inst = placement.instance
    cid = inst.cluster_id
    n_groups = int(cid.max()) if cid.size else 0
    total = 0
    split = []
    for g in range(1, n_groups + 1):
        members = np.nonzero(cid == g)[0]
        components = connected_components(placement.xywh, members)
        if components > 1:
            total += components - 1
            split.append(g)
    return total, split


def check_mib(placement: Placement) -> tuple[int, list[int]]:
    """``V_mib = sum_q (s_q - 1)``; returns ``(violations, broken MIB group ids)``."""
    inst = placement.instance
    mid = inst.mib_id
    n_groups = int(mid.max()) if mid.size else 0
    total = 0
    broken = []
    for g in range(1, n_groups + 1):
        members = np.nonzero(mid == g)[0]
        if len(members) <= 1:
            continue
        distinct = {
            (round(float(placement.w[m]), 4), round(float(placement.h[m]), 4))
            for m in members
        }
        if len(distinct) > 1:
            total += len(distinct) - 1
            broken.append(g)
    return total, broken


# Tolerance of "edge coordinate equals the bbox edge".
BOUNDARY_EPS = 1e-6


def check_boundary(placement: Placement) -> tuple[int, list[int]]:
    """``V_boundary``: the coded blocks missing a required bbox edge; returns ``(count, blocks)``.

    Codes are the bitmask 1=left 2=right 4=top 8=bottom; an edge is met when the block's
    coordinate equals the bbox coordinate within ``BOUNDARY_EPS``.
    """
    inst = placement.instance
    codes = inst.boundary_code
    coded = np.nonzero(codes != 0)[0]
    if coded.size == 0:
        return 0, []
    x_min, y_min, x_max, y_max = bounding_box(placement.xywh)
    eps = BOUNDARY_EPS
    total = 0
    bad_blocks = []
    for i in coded:
        code = int(codes[i])
        bx, by, bw, bh = (float(v) for v in placement.xywh[i])
        touches = {
            1: abs(bx - x_min) < eps,
            2: abs(bx + bw - x_max) < eps,
            4: abs(by + bh - y_max) < eps,
            8: abs(by - y_min) < eps,
        }
        if not all(touches[bit] for bit in (1, 2, 4, 8) if code & bit):
            total += 1
            bad_blocks.append(int(i))
    return total, bad_blocks


def soft_denominator(inst: FloorplanInstance) -> int:
    """``N_soft = |B_boundary| + sum_p (|G_p| - 1) + sum_q (|M_q| - 1)``."""
    n_boundary = int((inst.boundary_code != 0).sum())
    cluster_sizes = np.bincount(inst.cluster_id[inst.cluster_id > 0])
    mib_sizes = np.bincount(inst.mib_id[inst.mib_id > 0])
    grouping_slots = int((cluster_sizes[cluster_sizes > 0] - 1).sum())
    mib_slots = int((mib_sizes[mib_sizes > 0] - 1).sum())
    return n_boundary + grouping_slots + mib_slots


def check_soft(placement: Placement) -> SoftReport:
    """Run all soft-constraint checks and assemble ``V_rel``."""
    grouping, split = check_grouping(placement)
    mib, broken = check_mib(placement)
    boundary, bad_blocks = check_boundary(placement)
    return SoftReport(
        grouping=grouping,
        mib=mib,
        boundary=boundary,
        boundary_blocks=bad_blocks,
        grouping_groups=split,
        mib_groups=broken,
        n_soft=soft_denominator(placement.instance),
    )
