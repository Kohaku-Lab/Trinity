"""Severity-graded soft-constraint score ``sev_rel`` and the continuous cost ``soft_cost``.

``sev_rel`` has the slots and the ``N_soft`` denominator of ``V_rel``, but each slot
contributes its severity ``tanh(d / tau)`` in ``[0, 1)``, with ``d`` the slot's distance in
``/ s`` units (the gap to the nearest cluster member, the log-shape deviation from the MIB
group mean, the distance to the required bbox edges).

``soft_cost = (1 + ALPHA (hpwl_gap + area_gap)) exp(BETA sev_rel) exp(BETA_OVERLAP overlap_ratio)``:
the per-case cost with ``sev_rel`` in place of ``V_rel`` and an overlap factor in place of
the infeasibility penalty.
"""

from dataclasses import dataclass

import numpy as np

from trinity.floorplan.geometry import bounding_box
from trinity.floorplan.geometry_primitives import EPS, pair_gap
from trinity.floorplan.scoring.cost import ALPHA, BETA
from trinity.floorplan.scoring.soft import soft_denominator
from trinity.floorplan.types import Placement

TAU_GROUP = 0.01
TAU_MIB = 0.05
TAU_BOUNDARY = 0.01
BETA_OVERLAP = 10.0


@dataclass
class SeverityScore:
    """Per-slot severities, their ``N_soft``-relative sum, and the continuous cost."""

    grouping: float
    mib: float
    boundary: float
    n_soft: int
    sev_rel: float
    soft_cost: float


def _slot(distance, tau: float):
    """Severity of one violated slot: ``tanh(distance / tau)``, in ``[0, 1)``."""
    return np.tanh(np.asarray(distance, dtype=np.float64) / tau)


def _drop_smallest(values: np.ndarray) -> float:
    """The sum of ``values`` without its smallest entry."""
    if values.size <= 1:
        return 0.0
    return float(values.sum() - values.min())


def group_severity(placement: Placement, tau: float = TAU_GROUP) -> float:
    """Summed severity of each cluster member's gap to its nearest same-cluster member, ``/s``."""
    inst = placement.instance
    cluster = inst.cluster_id
    xywh = placement.xywh
    total = 0.0
    for g in np.unique(cluster[cluster > 0]):
        members = np.nonzero(cluster == g)[0]
        if members.size < 2:
            continue
        gaps = np.array(
            [
                min(pair_gap(xywh[i], xywh[j]) for j in members if j != i)
                for i in members
            ]
        )
        total += _drop_smallest(_slot(gaps / (inst.s + EPS), tau))
    return total


def mib_severity(placement: Placement, tau: float = TAU_MIB) -> float:
    """Summed severity of each MIB member's log-shape deviation from its group mean."""
    inst = placement.instance
    mib = inst.mib_id
    total = 0.0
    for g in np.unique(mib[mib > 0]):
        members = np.nonzero(mib == g)[0]
        if members.size < 2:
            continue
        log_w = np.log(np.maximum(placement.xywh[members, 2], EPS))
        log_h = np.log(np.maximum(placement.xywh[members, 3], EPS))
        deviation = np.abs(log_w - log_w.mean()) + np.abs(log_h - log_h.mean())
        total += _drop_smallest(_slot(deviation, tau))
    return total


def boundary_severity(placement: Placement, tau: float = TAU_BOUNDARY) -> float:
    """Summed severity of each coded block's distance from its required bbox edge, ``/s``."""
    inst = placement.instance
    codes = inst.boundary_code
    coded = np.nonzero(codes != 0)[0]
    if coded.size == 0:
        return 0.0
    x_min, y_min, x_max, y_max = bounding_box(placement.xywh)
    total = 0.0
    for i in coded:
        code = int(codes[i])
        x, y, w, h = placement.xywh[i]
        distance = 0.0
        if code & 1:
            distance += abs(x - x_min)
        if code & 2:
            distance += abs(x + w - x_max)
        if code & 4:
            distance += abs(y + h - y_max)
        if code & 8:
            distance += abs(y - y_min)
        total += float(_slot(distance / (inst.s + EPS), tau))
    return total


def score_severity(
    placement: Placement, hpwl_gap: float, area_gap: float, overlap_ratio: float = 0.0
) -> SeverityScore:
    """``sev_rel`` and ``soft_cost`` of ``placement`` from its gaps and ``overlap_ratio``."""
    grouping = group_severity(placement)
    mib = mib_severity(placement)
    boundary = boundary_severity(placement)
    n_soft = soft_denominator(placement.instance)
    sev_rel = (grouping + mib + boundary) / max(n_soft, 1)
    quality = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    soft_cost = quality * np.exp(BETA * sev_rel) * np.exp(BETA_OVERLAP * overlap_ratio)
    return SeverityScore(
        grouping=grouping,
        mib=mib,
        boundary=boundary,
        n_soft=n_soft,
        sev_rel=float(sev_rel),
        soft_cost=float(soft_cost),
    )
