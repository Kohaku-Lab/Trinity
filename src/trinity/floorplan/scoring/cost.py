"""The per-case cost of FloorSet and the scorer backends.

HPWL is the weighted center-to-center Manhattan length summed over the b2b and p2b nets
(the FloorSet convention). A case is feasible iff its hard constraints hold; an
infeasible case costs ``M_PENALTY``. The cost of a feasible case is ``(1 + ALPHA
(hpwl_gap + area_gap)) exp(BETA V_rel)``.

Scorers:

* ``full`` -- hard feasibility (including immutability) and the soft term ``V_rel``;
* ``full_fast`` -- the same result, with the HPWL edge data cached per
  instance and the sum vectorized in the same order (bitwise-identical);
* ``stub`` -- overlap and area feasibility only, ``V_rel = 0``.
"""

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np

from trinity.floorplan.geometry import bbox_area
from trinity.floorplan.registry import SCORER
from trinity.floorplan.scoring.feasibility import FeasibilityReport, check_feasibility
from trinity.floorplan.scoring.soft import SoftReport, check_soft
from trinity.floorplan.types import FloorplanInstance, Placement

ALPHA = 0.5
BETA = 2.0
M_PENALTY = 10.0
GAP_FLOOR = 1e-6

METRIC_AREA = 0
METRIC_B2B_WL = 6
METRIC_P2B_WL = 7


@dataclass
class CaseScore:
    """The full scoring of one layout."""

    cost: float
    feasible: bool
    hpwl: float
    hpwl_gap: float
    area: float
    area_gap: float
    v_rel: float
    feasibility: FeasibilityReport
    soft: SoftReport | None
    block_count: int


def hpwl(placement: Placement) -> float:
    """Weighted centroid-to-centroid Manhattan length over b2b + p2b nets."""
    inst = placement.instance
    centers = placement.centers
    total = 0.0
    for edge in inst.b2b:
        if edge[0] == -1:
            continue
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i >= inst.block_count or j >= inst.block_count:
            continue
        total += w * (
            abs(centers[i, 0] - centers[j, 0]) + abs(centers[i, 1] - centers[j, 1])
        )
    for edge in inst.p2b:
        if edge[0] == -1:
            continue
        p, b, w = int(edge[0]), int(edge[1]), float(edge[2])
        if p >= inst.pins_pos.shape[0] or b >= inst.block_count:
            continue
        px, py = inst.pins_pos[p]
        total += w * (abs(px - centers[b, 0]) + abs(py - centers[b, 1]))
    return float(total)


@dataclass(frozen=True)
class _HPWLPlan:
    """The edge data of one instance for :func:`hpwl_fast`.

    Rows keep their source order with the ``-1`` and out-of-range rows dropped, as in
    :func:`hpwl`; weights are float64.
    """

    b_i: np.ndarray
    b_j: np.ndarray
    b_w: np.ndarray
    p_block: np.ndarray
    p_xy: np.ndarray
    p_w: np.ndarray


_HPWL_PLAN_CACHE: dict[int, tuple[weakref.ReferenceType, _HPWLPlan]] = {}
_FAST_HPWL_CONTEXT: ContextVar[bool] = ContextVar("fast_hpwl_context", default=False)


@contextmanager
def use_fast_hpwl(enabled: bool = True):
    """Make every ``full`` scoring inside the block
    use :func:`hpwl_fast` (context-local)."""
    token = _FAST_HPWL_CONTEXT.set(bool(enabled))
    try:
        yield
    finally:
        _FAST_HPWL_CONTEXT.reset(token)


def _build_hpwl_plan(inst: FloorplanInstance) -> _HPWLPlan:
    """The :class:`_HPWLPlan` of ``inst``."""
    n = inst.block_count
    b2b = np.asarray(inst.b2b).reshape(-1, 3)
    b2b = b2b[b2b[:, 0] != -1]
    b_i = b2b[:, 0].astype(np.intp, copy=False)
    b_j = b2b[:, 1].astype(np.intp, copy=False)
    b_w = b2b[:, 2].astype(np.float64, copy=False)
    b_valid = (b_i < n) & (b_j < n)

    p2b = np.asarray(inst.p2b).reshape(-1, 3)
    p2b = p2b[p2b[:, 0] != -1]
    p_pin = p2b[:, 0].astype(np.intp, copy=False)
    p_block = p2b[:, 1].astype(np.intp, copy=False)
    p_w = p2b[:, 2].astype(np.float64, copy=False)
    n_pins = inst.pins_pos.shape[0]
    p_valid = (p_pin < n_pins) & (p_block < n)
    p_pin, p_block, p_w = p_pin[p_valid], p_block[p_valid], p_w[p_valid]
    p_xy = (
        np.asarray(inst.pins_pos)[p_pin]
        if p_pin.size
        else np.empty((0, 2), dtype=np.float64)
    )

    return _HPWLPlan(
        b_i=np.ascontiguousarray(b_i[b_valid]),
        b_j=np.ascontiguousarray(b_j[b_valid]),
        b_w=np.ascontiguousarray(b_w[b_valid]),
        p_block=np.ascontiguousarray(p_block),
        p_xy=np.ascontiguousarray(p_xy, dtype=np.float64),
        p_w=np.ascontiguousarray(p_w),
    )


def _hpwl_plan(inst: FloorplanInstance) -> _HPWLPlan:
    """The cached :class:`_HPWLPlan` of ``inst``
    (keyed by ``id`` with a weak reference)."""
    key = id(inst)
    cached = _HPWL_PLAN_CACHE.get(key)
    if cached is not None and cached[0]() is inst:
        return cached[1]
    plan = _build_hpwl_plan(inst)
    try:
        _HPWL_PLAN_CACHE[key] = (weakref.ref(inst), plan)
    except TypeError:
        pass
    return plan


def _accumulate_from_zero(values: np.ndarray) -> np.generic:
    """The sequential sum ``0.0 + values[0] + values[1] + ...`` in array order."""
    ordered = np.empty(values.size + 1, dtype=values.dtype)
    ordered[0] = 0.0
    ordered[1:] = values
    return np.add.accumulate(ordered, dtype=values.dtype)[-1]


def _hpwl_with_plan(xywh: np.ndarray, plan: _HPWLPlan) -> float:
    """The HPWL of ``xywh`` under ``plan``, summed in edge order."""
    centers = xywh[:, :2] + xywh[:, 2:] / 2.0
    b_values = None
    if plan.b_i.size:
        b_dist = np.abs(centers[plan.b_i, 0] - centers[plan.b_j, 0]) + np.abs(
            centers[plan.b_i, 1] - centers[plan.b_j, 1]
        )
        b_values = plan.b_w.astype(b_dist.dtype, copy=False) * b_dist
    p_values = None
    if plan.p_block.size:
        p_dist = np.abs(plan.p_xy[:, 0] - centers[plan.p_block, 0]) + np.abs(
            plan.p_xy[:, 1] - centers[plan.p_block, 1]
        )
        p_values = plan.p_w.astype(p_dist.dtype, copy=False) * p_dist

    total = None
    if b_values is not None:
        total = _accumulate_from_zero(b_values)
    if p_values is not None:
        if total is None:
            total = _accumulate_from_zero(p_values)
        else:
            dtype = np.result_type(np.asarray(total).dtype, p_values.dtype)
            ordered = np.empty(p_values.size + 1, dtype=dtype)
            ordered[0] = total
            ordered[1:] = p_values
            total = np.add.accumulate(ordered, dtype=dtype)[-1]
    return 0.0 if total is None else float(total)


def hpwl_fast(placement: Placement) -> float:
    """Vectorized :func:`hpwl` over a cached edge plan (bitwise-identical result)."""
    return _hpwl_with_plan(np.asarray(placement.xywh), _hpwl_plan(placement.instance))


def gap_baselines(placement: Placement, fast_hpwl: bool = False) -> tuple[float, float]:
    """``(hpwl_baseline, area_baseline)`` from the dataset metrics.

    Without metrics, the baselines are recomputed on the ground
    truth (or on ``placement`` itself when there is none).
    """
    inst = placement.instance
    hpwl_fn = hpwl_fast if fast_hpwl else hpwl
    hpwl_base = hpwl_fn(placement) if inst.gt_positions is None else None
    area_base = None
    if inst.metrics is not None and len(inst.metrics) >= 8:
        m = inst.metrics
        if m[METRIC_AREA] > 0:
            area_base = float(m[METRIC_AREA])
        if m[METRIC_B2B_WL] > 0 and m[METRIC_P2B_WL] >= 0:
            hpwl_base = float(m[METRIC_B2B_WL] + m[METRIC_P2B_WL])
    if hpwl_base is None:
        gt = Placement(xywh=inst.gt_positions, instance=inst)
        hpwl_base = hpwl_fn(gt)
    if area_base is None:
        area_base = (
            bbox_area(inst.gt_positions)
            if inst.gt_positions is not None
            else bbox_area(placement.xywh)
        )
    return hpwl_base, area_base


def compute_cost(
    hpwl_gap: float, area_gap: float, v_rel: float, feasible: bool
) -> float:
    """The per-case cost ``(1 + ALPHA (hpwl_gap + area_gap)) exp(BETA V_rel)``.

    Gaps are clamped at 0; an infeasible case
    costs ``M_PENALTY``; the runtime factor is 1.
    """
    if not feasible:
        return M_PENALTY
    quality = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    return quality * float(np.exp(BETA * v_rel))


def _score(
    placement: Placement,
    use_soft: bool,
    enforce_immutability: bool,
    fast_hpwl: bool = False,
) -> CaseScore:
    """Score one layout: feasibility, HPWL / area gaps, optional ``V_rel``, cost."""
    inst = placement.instance
    feas = check_feasibility(placement)
    feasible = feas.is_feasible(enforce_immutability=enforce_immutability)

    layout_hpwl = hpwl_fast(placement) if fast_hpwl else hpwl(placement)
    layout_area = bbox_area(placement.xywh)
    hpwl_base, area_base = gap_baselines(placement, fast_hpwl=fast_hpwl)
    hpwl_gap = (layout_hpwl - hpwl_base) / max(hpwl_base, GAP_FLOOR)
    area_gap = (layout_area - area_base) / max(area_base, GAP_FLOOR)

    soft = check_soft(placement) if use_soft else None
    v_rel = soft.v_rel if soft is not None else 0.0

    return CaseScore(
        cost=compute_cost(hpwl_gap, area_gap, v_rel, feasible),
        feasible=feasible,
        hpwl=layout_hpwl,
        hpwl_gap=hpwl_gap,
        area=layout_area,
        area_gap=area_gap,
        v_rel=v_rel,
        feasibility=feas,
        soft=soft,
        block_count=inst.block_count,
    )


@SCORER.register("full")
def score_full(placement: Placement) -> CaseScore:
    """Hard feasibility (including immutability) + the soft term ``exp(BETA V_rel)``.

    Uses :func:`hpwl_fast` inside a :func:`use_fast_hpwl` block.
    """
    return _score(
        placement,
        use_soft=True,
        enforce_immutability=True,
        fast_hpwl=_FAST_HPWL_CONTEXT.get(),
    )


@SCORER.register("full_fast")
def score_full_fast(placement: Placement) -> CaseScore:
    """The ``full`` score, always computed with :func:`hpwl_fast`."""
    return _score(placement, use_soft=True, enforce_immutability=True, fast_hpwl=True)


@SCORER.register("stub")
def score_stub(placement: Placement) -> CaseScore:
    """Overlap and area feasibility only, ``V_rel = 0``."""
    return _score(placement, use_soft=False, enforce_immutability=False)
