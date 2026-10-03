"""Continuous metrics of a layout, defined before legalization.

Each metric is a ``Placement -> float`` adapter over a primitive in
:mod:`trinity.floorplan.geometry_primitives`, normalized by ``s = sqrt(sum area)``:

  overlap_ratio   pairwise intersection area / total block area
  compactness     bbox area / total block area (>= 1)
  wl_norm         weighted Manhattan net length / (s * total net weight)
  boundary_dist   mean distance of the coded blocks to their required edges / s
  group_gap       mean gap from each clustered block to its nearest cluster member / s
  mib_var         mean within-MIB-group log-shape variance
  fixed_dist      mean fixed / preplaced deviation from target
  score_pre       a weighted sum of the above (overlap weighted ``W_OVERLAP``)
"""

from dataclasses import dataclass

import numpy as np

from trinity.floorplan import geometry_primitives as gp
from trinity.floorplan.registry import SCORER
from trinity.floorplan.scoring.cost import hpwl
from trinity.floorplan.types import Placement

EPS = gp.EPS
W_OVERLAP = 10.0
W_BOUNDARY = 1.0
W_GROUP = 1.0
W_MIB = 0.5
W_FIXED = 1.0
W_COMPACT = 0.5


@dataclass
class ContinuousScore:
    """The continuous metrics of one layout (see the module docstring)."""

    overlap_ratio: float
    compactness: float
    wl_norm: float
    boundary_dist: float
    group_gap: float
    mib_var: float
    fixed_dist: float
    score_pre: float
    block_count: int


def overlap_ratio(xywh: np.ndarray) -> float:
    """Pairwise intersection area / total block area."""
    return gp.overlap_ratio(xywh)


def compactness(xywh: np.ndarray) -> float:
    """Bbox area / total block area."""
    return gp.bbox_compactness(xywh)


def wl_norm(placement: Placement) -> float:
    """HPWL / (s * total net weight)."""
    inst = placement.instance
    total_weight = 0.0
    for edge in inst.b2b:
        if (
            edge[0] != -1
            and int(edge[0]) < inst.block_count
            and int(edge[1]) < inst.block_count
        ):
            total_weight += float(edge[2])
    for edge in inst.p2b:
        valid_pin = int(edge[0]) < inst.pins_pos.shape[0]
        if edge[0] != -1 and valid_pin and int(edge[1]) < inst.block_count:
            total_weight += float(edge[2])
    return float(hpwl(placement) / (inst.s * (total_weight + EPS)))


def boundary_dist(placement: Placement) -> float:
    """Mean distance of the boundary-coded blocks to their required bbox edges / s."""
    inst = placement.instance
    return gp.boundary_distance(placement.xywh, inst.boundary_code, inst.s)


def group_gap(placement: Placement) -> float:
    """Mean gap from each clustered block to its nearest cluster member / s."""
    inst = placement.instance
    return gp.group_gap(placement.xywh, inst.cluster_id, inst.s)


def mib_var(placement: Placement) -> float:
    """Mean within-MIB-group variance of the log shape ``(log w, log h)``."""
    inst = placement.instance
    return gp.mib_log_shape_var(placement.xywh, inst.mib_id)


def fixed_dist(placement: Placement) -> float:
    """Mean fixed / preplaced deviation from target (center / s + shape / sqrt(area))."""
    inst = placement.instance
    return gp.fixed_deviation(
        placement.xywh,
        inst.target_positions,
        inst.area_targets,
        inst.is_fixed,
        inst.is_preplaced,
        inst.s,
    )


@SCORER.register("continuous")
def score_continuous(placement: Placement) -> ContinuousScore:
    """Every continuous metric of ``placement`` and their weighted sum ``score_pre``."""
    xywh = placement.xywh
    overlap = overlap_ratio(xywh)
    compact = compactness(xywh)
    wirelength = wl_norm(placement)
    boundary = boundary_dist(placement)
    gap = group_gap(placement)
    mib = mib_var(placement)
    fixed = fixed_dist(placement)
    score_pre = (
        wirelength
        + W_COMPACT * compact
        + W_OVERLAP * overlap
        + W_BOUNDARY * boundary
        + W_GROUP * gap
        + W_MIB * mib
        + W_FIXED * fixed
    )
    return ContinuousScore(
        overlap_ratio=overlap,
        compactness=compact,
        wl_norm=wirelength,
        boundary_dist=boundary,
        group_gap=gap,
        mib_var=mib,
        fixed_dist=fixed,
        score_pre=float(score_pre),
        block_count=placement.instance.block_count,
    )
