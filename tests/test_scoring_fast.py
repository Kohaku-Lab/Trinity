"""The fast HPWL path and the ``full_fast``
scorer equal the reference scorer exactly."""

import numpy as np

import trinity.floorplan.legalize  # noqa: F401  (register: scorers)
from trinity.floorplan.geometry import bbox_area
from trinity.floorplan.registry import SCORER, resolve
from trinity.floorplan.scoring import use_fast_hpwl
from trinity.floorplan.scoring.cost import hpwl, hpwl_fast
from trinity.floorplan.types import FloorplanInstance, Placement


def _instance(xywh, *, b2b=None, p2b=None, pins=None, metrics=None):
    n = len(xywh)
    gt = xywh.copy()
    return FloorplanInstance(
        block_count=n,
        area_targets=gt[:, 2] * gt[:, 3],
        constraints=np.zeros((n, 8), dtype=np.float64),
        b2b=np.empty((0, 3)) if b2b is None else np.asarray(b2b, dtype=np.float64),
        p2b=np.empty((0, 3)) if p2b is None else np.asarray(p2b, dtype=np.float64),
        pins_pos=(
            np.empty((0, 2)) if pins is None else np.asarray(pins, dtype=np.float64)
        ),
        target_positions=np.full((n, 4), -1.0),
        gt_positions=gt,
        metrics=(
            np.array([bbox_area(gt), 0, 0, 0, 0, 0, 0, 0])
            if metrics is None
            else metrics
        ),
    )


def _sample():
    xywh = np.array(
        [[0.0, 0.0, 2.0, 1.0], [3.0, 0.0, 1.0, 2.0], [1.0, 2.5, 1.5, 1.0]],
        dtype=np.float64,
    )
    b2b = [[0, 1, 1.0], [1, 2, 2.0], [0, 2, 0.5], [-1, -1, 0.0]]  # a sentinel row too
    p2b = [[0, 2, 1.5], [1, 0, 0.5]]
    pins = [[5.0, 5.0], [-1.0, 2.0]]
    return _instance(xywh, b2b=b2b, p2b=p2b, pins=pins)


def test_fast_hpwl_bitwise_identical():
    inst = _sample()
    p = Placement(inst.gt_positions.copy(), inst)
    assert hpwl_fast(p) == hpwl(p)  # exact, not approximate


def test_full_fast_scorer_matches_full():
    inst = _sample()
    p = Placement(inst.gt_positions.copy(), inst)
    full = resolve("full", SCORER)(p)
    fast = resolve("full_fast", SCORER)(p)
    assert fast.cost == full.cost
    assert fast.hpwl == full.hpwl
    assert fast.feasible == full.feasible


def test_use_fast_hpwl_context_switches_full_backend():
    inst = _sample()
    p = Placement(inst.gt_positions.copy(), inst)
    base = resolve("full", SCORER)(p).cost
    with use_fast_hpwl():
        ctx = resolve("full", SCORER)(p).cost
    assert ctx == base  # exact fast path == legacy, so the wrapped result is unchanged


def test_fast_hpwl_no_nets_is_zero():
    inst = _instance(np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float64))
    p = Placement(inst.gt_positions.copy(), inst)
    assert hpwl_fast(p) == hpwl(p) == 0.0
