"""Descriptor invariances and the distribution metrics' sanity properties."""

import numpy as np
import pytest

from trinity.floorplan.scoring.descriptors import (
    DENSITY_DIM,
    POSITION_DIM,
    density_descriptor,
    position_descriptor,
)
from trinity.floorplan.scoring.distribution import frechet, median_bandwidth, mmd2
from trinity.floorplan.types import FloorplanInstance


def _instance(n=12, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.uniform(1.0, 4.0, size=n)
    wh = np.sqrt(area)
    xy = rng.uniform(0.0, 10.0, size=(n, 2))
    cons = np.zeros((n, 5))
    cons[:3, 0] = 1
    cons[3, 1] = 1
    b2b = np.column_stack(
        [rng.integers(0, n, 20), rng.integers(0, n, 20), rng.uniform(0.5, 2, 20)]
    ).astype(float)
    inst = FloorplanInstance(
        block_count=n,
        area_targets=area,
        constraints=cons,
        b2b=b2b,
        p2b=np.zeros((0, 3)),
        pins_pos=np.zeros((0, 2)),
        target_positions=-np.ones((n, 4)),
    )
    return np.column_stack([xy, wh, wh]), inst


def test_shapes():
    xywh, inst = _instance()
    assert density_descriptor(xywh, inst, G=8).shape == (DENSITY_DIM[8],)
    assert density_descriptor(xywh, inst, G=12, frame="fixed").shape == (
        DENSITY_DIM[12],
    )
    assert position_descriptor(xywh, inst).shape == (POSITION_DIM,)


def test_permutation_and_translation_invariance():
    xywh, inst = _instance()
    perm = np.random.default_rng(1).permutation(inst.block_count)
    inst_p = FloorplanInstance(
        block_count=inst.block_count,
        area_targets=inst.area_targets[perm],
        constraints=inst.constraints[perm],
        b2b=inst.b2b.copy(),
        p2b=inst.p2b,
        pins_pos=inst.pins_pos,
        target_positions=inst.target_positions[perm],
    )
    inv = np.argsort(perm)
    inst_p.b2b[:, 0] = inv[inst.b2b[:, 0].astype(int)]
    inst_p.b2b[:, 1] = inv[inst.b2b[:, 1].astype(int)]
    for frame in ("bbox", "fixed"):
        d0 = density_descriptor(xywh, inst, frame=frame)
        assert np.allclose(density_descriptor(xywh[perm], inst_p, frame=frame), d0)
        shifted = xywh + np.array([3.0, -2.0, 0.0, 0.0])
        assert np.allclose(density_descriptor(shifted, inst, frame=frame), d0)
    p0 = position_descriptor(xywh, inst)
    assert np.allclose(position_descriptor(xywh[perm], inst_p), p0)
    assert np.allclose(
        position_descriptor(xywh + np.array([3.0, -2.0, 0, 0]), inst), p0
    )


def test_density_fields_sum_to_area_fraction():
    xywh, inst = _instance()
    G = 8
    d = density_descriptor(xywh, inst, G=G, frame="bbox")
    x0, y0 = xywh[:, 0].min(), xywh[:, 1].min()
    w = (xywh[:, 0] + xywh[:, 2]).max() - x0
    h = (xywh[:, 1] + xywh[:, 3]).max() - y0
    all_field = d[: G * G]
    assert all_field.sum() * (w / G) * (h / G) == pytest.approx(
        (xywh[:, 2] * xywh[:, 3]).sum(), rel=1e-6
    )
    assert d[3 * G * G :].sum() == pytest.approx(
        1.0
    )  # net-weight field is a distribution


def test_frechet_and_mmd_zero_on_identical_sets_and_positive_on_shifted():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 6))
    b = a.copy()
    c = a + 1.0
    assert frechet(a, b)[0] == pytest.approx(0.0, abs=1e-8)
    assert frechet(a, c)[0] > 5.0
    bw = median_bandwidth(a)
    assert abs(mmd2(a[:250], a[250:], bw)) < 0.02
    assert mmd2(a, c, bw) > 0.1
