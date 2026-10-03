"""Wire dropout: rate mixture, row dropping, and the recomputed HPWL baseline."""

import numpy as np
import pytest

import trinity  # noqa: F401  (register all)
from trinity.augment_ops import WireDropoutOp, build_pipeline
from trinity.floorplan.scoring.cost import METRIC_B2B_WL, METRIC_P2B_WL, hpwl_fast
from trinity.floorplan.types import FloorplanInstance, Placement


def _instance(n=8, e_b=20, e_p=6, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.uniform(1.0, 4.0, size=n)
    wh = np.sqrt(area)
    xy = rng.uniform(0.0, 10.0, size=(n, 2))
    gt = np.column_stack([xy, wh, wh])
    b2b = np.column_stack(
        [rng.integers(0, n, e_b), rng.integers(0, n, e_b), rng.uniform(0.5, 2, e_b)]
    ).astype(np.float64)
    p2b = np.column_stack(
        [rng.integers(0, 4, e_p), rng.integers(0, n, e_p), rng.uniform(0.5, 2, e_p)]
    ).astype(np.float64)
    inst = FloorplanInstance(
        block_count=n,
        area_targets=area,
        constraints=np.zeros((n, 5)),
        b2b=b2b,
        p2b=p2b,
        pins_pos=rng.uniform(0.0, 10.0, size=(4, 2)),
        target_positions=-np.ones((n, 4)),
        gt_positions=gt,
        metrics=np.zeros(8),
    )
    inst.metrics[0] = 100.0
    inst.metrics[METRIC_B2B_WL] = 1.0
    return inst


def test_zero_rate_is_identity():
    inst = _instance()
    out = WireDropoutOp(keep=1.0, packing=0.0).apply(inst, np.random.default_rng(0))
    assert out is inst


def test_full_dropout_removes_every_wire_and_zeroes_baseline():
    inst = _instance()
    op = WireDropoutOp(keep=0.0, packing=1.0, packing_range=(1.0, 1.0))
    out = op.apply(inst, np.random.default_rng(1))
    assert out.b2b.shape[0] == 0 and out.p2b.shape[0] == 0
    assert out.metrics[METRIC_B2B_WL] == 0.0 and out.metrics[METRIC_P2B_WL] == 0.0
    assert inst.b2b.shape[0] == 20  # the source instance is untouched


def test_partial_dropout_recomputes_hpwl_on_reduced_netlist():
    inst = _instance()
    op = WireDropoutOp(keep=0.0, packing=0.0, beta=(50.0, 50.0))  # rate ~0.5
    out = op.apply(inst, np.random.default_rng(2))
    assert 0 < out.b2b.shape[0] < 20
    expect = hpwl_fast(Placement(xywh=inst.gt_positions, instance=out))
    assert out.metrics[METRIC_B2B_WL] == pytest.approx(expect)
    assert out.metrics[0] == 100.0  # untouched entries carried over


def test_rate_mixture_matches_branch_probabilities():
    op = WireDropoutOp(keep=0.5, packing=0.1, beta=(1.0, 3.0), packing_range=(0.9, 1.0))
    rng = np.random.default_rng(3)
    rates = np.array([op.rate(rng) for _ in range(20_000)])
    assert abs((rates == 0.0).mean() - 0.5) < 0.02
    assert abs((rates >= 0.9).mean() - 0.1) < 0.02
    mild = rates[(rates > 0.0) & (rates < 0.9)]
    assert abs(mild.mean() - 0.25) < 0.02


def test_pipeline_entry_builds():
    pipe = build_pipeline([{"wire_dropout": {"keep": 0.5, "packing": 0.1}}])
    assert isinstance(pipe.ops[0], WireDropoutOp)
