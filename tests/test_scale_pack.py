"""Gates of the ``scale_pack`` legalizer.

Overlap-free output on jittered legal layouts at the first rung, anchors and shapes untouched,
cluster contact and boundary codes met, the pinned squeeze and a tangled layout feasible
through the ladder, MIB shapes identical, and the meaning of the expansion factor.
"""

import numpy as np

import trinity.floorplan.legalize  # noqa: F401
from trinity.floorplan.geometry import connected_components, overlapping_pairs
from trinity.floorplan.legalize.scale_pack import expand, last_info, scale_pack_legalize
from trinity.floorplan.scoring import validate
from trinity.floorplan.scoring.soft import check_boundary, check_mib
from trinity.floorplan.types import FloorplanInstance, Placement


def _instance(
    xywh, preplaced=(), fixed=(), cluster=None, mib=None, boundary=None, areas=None
):
    n = len(xywh)
    cons = np.zeros((n, 5))
    tp = -np.ones((n, 4))
    for i in preplaced:
        cons[i, 1] = 1
        tp[i] = xywh[i]
    for i in fixed:
        cons[i, 0] = 1
        tp[i, 2:] = xywh[i, 2:]
    if cluster is not None:
        cons[:, 3] = cluster
    if mib is not None:
        cons[:, 2] = mib
    if boundary is not None:
        cons[:, 4] = boundary
    area = xywh[:, 2] * xywh[:, 3] if areas is None else np.asarray(areas, np.float64)
    return FloorplanInstance(
        block_count=n,
        area_targets=area,
        constraints=cons,
        b2b=np.zeros((0, 3)),
        p2b=np.zeros((0, 3)),
        pins_pos=np.zeros((0, 2)),
        target_positions=tp,
    )


def _grid_layout(rows, cols, seed, jitter=0.04):
    """Random-size blocks abutting in a row-major grid, centres jittered by ``jitter`` of the scale."""
    rng = np.random.default_rng(seed)
    n = rows * cols
    w, h = rng.uniform(1.0, 3.0, n), rng.uniform(1.0, 3.0, n)
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
    s = np.sqrt((w * h).sum())
    xywh[:, :2] += rng.normal(0.0, jitter * s, (n, 2))
    return xywh


def _tangled_layout(n, seed, spread=4.0):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(0, spread, n),
            rng.uniform(0, spread, n),
            rng.uniform(1, 3, n),
            rng.uniform(1, 3, n),
        ]
    )


def test_jittered_layouts_become_feasible_at_the_first_rung():
    for seed in range(5):
        xywh = _grid_layout(5, 6, seed)
        inst = _instance(xywh)
        assert overlapping_pairs(xywh)
        out = scale_pack_legalize(Placement(xywh, inst))
        assert not overlapping_pairs(out.xywh)
        assert validate(out, "full_fast").score.feasible
        assert np.allclose(out.xywh[:, 2] * out.xywh[:, 3], inst.area_targets)
        assert last_info()["rung"] == 1 and last_info()["lam"] > 1.0


def test_pinned_blocks_bit_exact_and_fixed_shapes_kept():
    xywh = _grid_layout(4, 6, 7)
    inst = _instance(xywh, preplaced=(0, 9, 14), fixed=(3, 4))
    out = scale_pack_legalize(Placement(xywh, inst))
    assert not overlapping_pairs(out.xywh)
    assert np.array_equal(out.xywh[[0, 9, 14]], xywh[[0, 9, 14]])
    assert np.array_equal(out.xywh[[3, 4], 2:], xywh[[3, 4], 2:])
    assert validate(out, "full_fast").score.feasible


def test_cluster_members_end_connected_and_boundary_codes_met():
    xywh = _grid_layout(4, 5, 3)
    cluster = np.zeros(20)
    cluster[[6, 7, 11, 12]] = 1
    cluster[[13, 14]] = 2
    boundary = np.zeros(20)
    boundary[0], boundary[4], boundary[19], boundary[15] = 9, 8, 6, 1
    inst = _instance(xywh, cluster=cluster, boundary=boundary)
    assert overlapping_pairs(xywh)
    out = scale_pack_legalize(Placement(xywh, inst))
    score = validate(out, "full_fast").score
    assert score.feasible
    assert last_info()["rung"] == 1
    assert connected_components(out.xywh, np.array([6, 7, 11, 12])) == 1
    assert connected_components(out.xywh, np.array([13, 14])) == 1
    assert check_boundary(out)[0] == 0
    assert score.v_rel == 0.0


def test_tangled_layout_stays_feasible_through_the_ladder():
    xywh = _tangled_layout(20, 3)
    cluster = np.zeros(20)
    cluster[:4] = 1
    inst = _instance(xywh, cluster=cluster)
    out = scale_pack_legalize(Placement(xywh, inst))
    assert validate(out, "full_fast").score.feasible
    assert 1 <= last_info()["rung"] <= 2


def test_block_squeezed_between_two_pinned_blocks_stays_feasible():
    xywh = np.array(
        [
            [0.0, 0.0, 2.0, 2.0],
            [3.0, 0.0, 2.0, 2.0],
            [1.5, 0.0, 2.0, 2.0],
            [0.0, 3.0, 1.0, 1.0],
        ]
    )
    inst = _instance(xywh, preplaced=(0, 1))
    out = scale_pack_legalize(Placement(xywh, inst))
    assert validate(out, "full_fast").score.feasible
    assert np.array_equal(out.xywh[:2], xywh[:2])
    assert last_info()["rung"] >= 1


def test_mib_group_members_share_one_shape():
    xywh = _grid_layout(3, 4, 5)
    areas = xywh[:, 2] * xywh[:, 3]
    areas[:3] = 4.0
    mib = np.zeros(12)
    mib[:3] = 1
    inst = _instance(xywh, mib=mib, areas=areas)
    out = scale_pack_legalize(Placement(xywh, inst))
    assert validate(out, "full_fast").score.feasible
    assert check_mib(out)[0] == 0


def test_aspect_pass_keeps_feasibility_areas_and_never_raises_the_cost():
    for seed in range(4):
        xywh = _grid_layout(4, 5, 10 + seed)
        cluster = np.zeros(20)
        cluster[[6, 7, 11, 12]] = 1
        boundary = np.zeros(20)
        boundary[0], boundary[4], boundary[19] = 9, 8, 6
        inst = _instance(xywh, cluster=cluster, boundary=boundary, fixed=(2,))
        base = scale_pack_legalize(Placement(xywh, inst), reshape=False)
        out = scale_pack_legalize(Placement(xywh, inst), reshape=True)
        sb, so = validate(base, "full_fast").score, validate(out, "full_fast").score
        assert so.feasible and so.cost <= sb.cost + 1e-12
        assert np.allclose(out.xywh[:, 2] * out.xywh[:, 3], inst.area_targets)
        assert np.array_equal(out.xywh[2, 2:], xywh[2, 2:])


def test_expansion_factor_grows_with_penetration_and_is_one_without_overlap():
    tight = np.array([[0.0, 0.0, 2.0, 2.0], [1.8, 0.0, 2.0, 2.0]])
    loose = np.array([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 2.0, 2.0]])
    apart = np.array([[0.0, 0.0, 2.0, 2.0], [2.5, 0.0, 2.0, 2.0]])
    assert 1.0 < expand(tight)[1] < expand(loose)[1]
    assert expand(apart)[1] == 1.0
    assert not overlapping_pairs(expand(loose)[0])
