"""Gates of the PARSAC solver: the instance-to-engine mapping, the B*-tree read from a legal layout
(every block once, root at the bottom-left, the engine's packing of that tree reproducing an abutting
grid), and a short anneal returning overlap-free checkpoints; the engine-bound gates skip without the
PARSAC source tree."""

import numpy as np
import pytest

import trinity_baselines.classical  # noqa: F401
from trinity.floorplan.geometry import overlapping_pairs
from trinity.floorplan.registry import SOLVER, build
from trinity.floorplan.types import FloorplanInstance, Placement
from trinity_baselines.classical.parsac import (
    compact,
    engine_available,
    to_engine_problem,
    tree_from_layout,
)


def _grid_instance(
    rows, cols, seed, preplaced=(), cluster=None, boundary=None, mib=None
):
    rng = np.random.default_rng(seed)
    n = rows * cols
    w, h = rng.integers(2, 6, n).astype(float), rng.integers(2, 6, n).astype(float)
    xywh = np.zeros((n, 4))
    y = 0.0
    for r in range(rows):
        x = 0.0
        row_h = h[r * cols : (r + 1) * cols].max()
        for c in range(cols):
            i = r * cols + c
            xywh[i] = [x, y, w[i], h[i]]
            x += w[i]
        y += row_h
    cons = np.zeros((n, 5))
    tp = -np.ones((n, 4))
    for i in preplaced:
        cons[i, 1] = 1
        tp[i] = xywh[i]
    if cluster is not None:
        cons[:, 3] = cluster
    if boundary is not None:
        cons[:, 4] = boundary
    if mib is not None:
        cons[:, 2] = mib
    pins = np.array([[0.0, 1.0], [x, y]])
    p2b = np.array([[0, 0, 1.0], [1, n - 1, 1.0]])
    b2b = np.array([[i, i + 1, 1.0] for i in range(n - 1)], dtype=float)
    inst = FloorplanInstance(
        block_count=n,
        area_targets=w * h,
        constraints=cons,
        b2b=b2b,
        p2b=p2b,
        pins_pos=pins,
        target_positions=tp,
        gt_positions=xywh.copy(),
    )
    return inst, xywh


def test_engine_problem_maps_codes_clusters_and_preplaced_blocks():
    cluster = np.zeros(12)
    cluster[[3, 4]] = 7
    cluster[[9, 10]] = 2
    boundary = np.zeros(12)
    boundary[0], boundary[11] = 9, 6
    inst, xywh = _grid_instance(
        3, 4, 1, preplaced=(5,), cluster=cluster, boundary=boundary
    )
    prob = to_engine_problem(inst, 1000.0)
    rows = np.array(prob["rows"])
    assert rows.shape == (12, 9)
    assert rows[0, 2] == 9 and rows[11, 2] == 6
    assert (
        sorted(set(rows[:, 3]) - {0}) == [1, 2]
        and rows[3, 3] == rows[4, 3] != rows[9, 3]
    )
    assert (
        rows[5, 6] == 1
        and rows[5, 5] == 1
        and rows[5, 7] == round(xywh[5, 0] * prob["scale"])
    )
    assert (rows[[i for i in range(12) if i != 5], 6] == 0).all()
    assert len(prob["nets"]) == 11 + 2 and prob["nets"][-1] == [11, 12 + 1]
    assert prob["canvas"][0] >= rows[:, 0].max()


def test_tree_from_layout_visits_every_block_once_from_the_bottom_left():
    for seed in range(4):
        _, xywh = _grid_instance(4, 5, seed)
        boxes = compact(np.rint(xywh * 10).astype(np.int64))
        edges = tree_from_layout(boxes)
        assert len(edges) == len(boxes) - 1
        children = [e[1] for e in edges]
        assert sorted(children) == sorted(set(range(len(boxes))) - {edges[0][0]})
        root = edges[0][0]
        assert boxes[root, 0] == 0 and boxes[root, 1] == 0


def test_compact_is_idempotent_on_an_abutting_grid():
    _, xywh = _grid_instance(3, 3, 2)
    boxes = np.rint(xywh * 10).astype(np.int64)
    once = compact(boxes)
    assert np.array_equal(compact(once), once)
    assert not overlapping_pairs(once.astype(float))


@pytest.mark.skipif(not engine_available(), reason="PARSAC source tree not present")
def test_anneal_returns_overlap_free_checkpoints_and_keeps_preplaced_blocks():
    inst, xywh = _grid_instance(3, 4, 3, preplaced=(0,))
    solver = build(
        {"name": "parsac", "budget": 4000, "checkpoint_stages": (0, 50)}, SOLVER
    )
    res = solver.anneal(inst, seed=1)
    assert len(res.checkpoints) == 3 and res.checkpoints[-1].label == "final"
    assert (
        res.checkpoints[0].steps < res.checkpoints[1].steps < res.checkpoints[2].steps
    )
    for cp in res.checkpoints:
        assert cp.xywh.shape == (12, 4)
        assert not overlapping_pairs(cp.xywh)
        assert np.allclose(cp.xywh[0, :2], xywh[0, :2], atol=1e-6)
    assert res.seconds > 0 and res.placement.xywh.shape == (12, 4)


@pytest.mark.skipif(not engine_available(), reason="PARSAC source tree not present")
def test_warm_start_packs_an_abutting_grid_back_onto_itself():
    inst, xywh = _grid_instance(3, 4, 4)
    solver = build(
        {"name": "parsac", "budget": 10, "checkpoint_stages": (0,), "ar_search": False},
        SOLVER,
    )
    res = solver.anneal(inst, seed=1, init=Placement(xywh=xywh.copy(), instance=inst))
    first = res.checkpoints[0].xywh
    assert not overlapping_pairs(first)
    assert np.allclose(first[:, 2:], xywh[:, 2:], atol=2.0 / res.info["scale"])
    assert res.info["warm_start"]
