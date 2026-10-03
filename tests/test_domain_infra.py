"""Problem-side correctness gates: data, parameterization, scoring, jitter, legalization, augment.

The data-dependent tests run against the FloorSet validation set and GSRC (resolved through
the project ``data/`` directory or ``TRINITY_FLOORSET`` / ``TRINITY_GSRC``) and are skipped
when the data is absent.
"""

import numpy as np
import pytest
import torch

import trinity.floorplan as K
from trinity.augment_ops import CondDropoutOp, build_pipeline
from trinity.data.dataset import _resolve, encode_instance
from trinity.floorplan.augment import random_transform, transform_instance
from trinity.floorplan.data import (
    ground_truth_placement,
    load_gsrc_case,
    load_validation_case,
)
from trinity.floorplan.data.paths import find_floorset_root, find_gsrc_root
from trinity.floorplan.jitter import apply_jitter
from trinity.floorplan.legalize import legalize
from trinity.floorplan.parameterize import xywh_to_z, z_to_xywh
from trinity.floorplan.scoring import score_continuous, validate
from trinity.floorplan.types import COL_PREPLACED, Placement
from trinity.registry import LATENT_PARAM, build


def _have(find_root) -> bool:
    try:
        find_root(allow_download=False)
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(
    not _have(find_floorset_root), reason="FloorSet validation set not present"
)
needs_gsrc = pytest.mark.skipif(
    not _have(find_gsrc_root), reason="GSRC benchmark not present"
)

JITTER_MODES = [
    "overlap",
    "area",
    "fixed",
    "preplaced",
    "grouping",
    "mib",
    "boundary",
    "noise",
]


def _gt_score(inst):
    return validate(Placement(inst.gt_positions.copy(), inst), "full").score


def test_registries_populated():
    assert set(K.SCORER.keys()) == {"full", "full_fast", "stub", "continuous"}
    assert "scale_pack" in K.LEGALIZER.keys()
    assert "placement" in K.RENDERER.keys()
    assert {"overlap", "grouping", "boundary", "noise"} <= set(K.JITTER.keys())


@needs_data
def test_parameterize_round_trip():
    inst = load_validation_case(21)
    gt = ground_truth_placement(inst)
    z = xywh_to_z(gt.xywh, inst.area_targets, inst.s)
    back = z_to_xywh(z, inst.area_targets, inst.s)
    c0 = gt.xywh[:, :2] + gt.xywh[:, 2:] / 2
    c1 = back[:, :2] + back[:, 2:] / 2
    assert np.abs(c0 - c1).max() < 1e-3
    assert np.abs(back[:, 2] * back[:, 3] - inst.area_targets).max() < 1e-3


@needs_data
def test_ground_truth_is_feasible_and_low_cost():
    for n in (21, 60, 120):
        gt = ground_truth_placement(load_validation_case(n))
        score = validate(gt, "full").score
        assert score.feasible
        assert score.hpwl_gap < 1e-3 and score.area_gap < 1e-3
        assert score.cost < 1.5


@needs_data
@pytest.mark.parametrize("mode", JITTER_MODES)
def test_jitter_is_detected(mode):
    gt = ground_truth_placement(load_validation_case(120))
    base = validate(gt, "full").score
    broken, _ = apply_jitter(gt, [mode], seed=7)
    after = validate(broken, "full").score
    # The jitter makes the layout infeasible or raises V_rel.
    assert (not after.feasible) or (after.v_rel > base.v_rel)


@needs_data
@pytest.mark.parametrize("mode", JITTER_MODES)
def test_legalizer_recovers_feasibility(mode):
    for n in (21, 120):
        gt = ground_truth_placement(load_validation_case(n))
        broken, _ = apply_jitter(gt, [mode], seed=7)
        result = legalize(broken, scorer="full")
        assert result.score.feasible, f"n={n} mode={mode} not legalized"


@needs_data
def test_stub_scorer_ignores_soft():
    gt = ground_truth_placement(load_validation_case(120))
    broken, _ = apply_jitter(gt, ["grouping"], seed=7)
    full = validate(broken, "full").score
    stub = validate(broken, "stub").score
    assert full.v_rel > 0
    assert stub.v_rel == 0.0


@needs_data
def test_continuous_scorer_ideal_on_gt_and_degrades():
    for n in (21, 60, 120):
        gt = ground_truth_placement(load_validation_case(n))
        sc = score_continuous(gt)
        assert sc.overlap_ratio < 1e-3
        assert 1.0 <= sc.compactness < 1.2
        assert sc.group_gap < 1e-3 and sc.fixed_dist < 1e-3
    gt = ground_truth_placement(load_validation_case(120))
    base = score_continuous(gt)
    broken, _ = apply_jitter(gt, ["overlap"], seed=7)
    after = score_continuous(broken)
    assert after.overlap_ratio > base.overlap_ratio
    assert after.score_pre > base.score_pre


@needs_data
@pytest.mark.parametrize("rot", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_augment_dihedral_preserves_cost(rot, flip):
    """Rotation / flip (+ a shift) leave cost, V_rel, HPWL and area unchanged."""
    inst = load_validation_case(60)
    base = _gt_score(inst)
    aug = _gt_score(transform_instance(inst, rot=rot, flip=flip, shift=(5.0, -3.0)))
    assert np.isclose(aug.cost, base.cost, atol=1e-3)
    assert np.isclose(aug.v_rel, base.v_rel, atol=1e-3)
    assert np.isclose(aug.hpwl, base.hpwl, atol=1e-2)
    assert np.isclose(aug.area, base.area, atol=1e-2)


@needs_data
def test_augment_cancelling_sequence_returns_to_identity():
    """A transform followed by its inverse (reverse order) returns the original case."""
    inst = load_validation_case(60)
    r, dx, dy = 3, 5.0, -3.0
    t = transform_instance(inst, rot=r, flip=True, shift=(dx, dy))
    u = transform_instance(t, shift=(-dx, -dy))
    u = transform_instance(u, flip=True)
    u = transform_instance(u, rot=(4 - r) % 4)
    assert np.array_equal(u.gt_positions, inst.gt_positions)
    assert np.array_equal(u.pins_pos, inst.pins_pos)
    assert np.array_equal(u.boundary_code, inst.boundary_code)
    assert np.array_equal(u.constraints, inst.constraints)


@needs_data
def test_augment_group_identities():
    """Four 90-degree turns and two flips are the identity."""
    inst = load_validation_case(60)
    f = inst
    for _ in range(4):
        f = transform_instance(f, rot=1)
    assert np.array_equal(f.gt_positions, inst.gt_positions)
    assert np.array_equal(f.pins_pos, inst.pins_pos)
    assert np.array_equal(f.boundary_code, inst.boundary_code)
    g = transform_instance(transform_instance(inst, flip=True), flip=True)
    assert np.array_equal(g.gt_positions, inst.gt_positions)
    assert np.array_equal(g.boundary_code, inst.boundary_code)


@needs_data
def test_random_transform_preserves_layout_objective():
    inst = load_validation_case(60)
    base = _gt_score(inst)
    rng = np.random.default_rng(7)
    for _ in range(20):
        aug = _gt_score(random_transform(inst, rng, shift_std=0.1))
        assert np.isclose(aug.hpwl, base.hpwl, atol=1e-2)
        assert np.isclose(aug.area, base.area, atol=1e-2)


@needs_data
def test_cond_dropout_drops_constraints_label_consistent():
    """cond_dropout zeroes the chosen groups without mutating the input or the label."""
    inst = load_validation_case(120)
    orig = inst.constraints.copy()
    base = _gt_score(inst)
    op = CondDropoutOp(
        boundary=1.0,
        cluster=1.0,
        mib=1.0,
        fixed=1.0,
        preplaced=1.0,
        cluster_per_group=False,
        mib_per_group=False,
    )
    d = op.apply(inst, np.random.default_rng(0))
    assert d is not inst and np.array_equal(inst.constraints, orig)
    assert (d.constraints == 0).all()
    assert (d.target_positions[orig[:, COL_PREPLACED] != 0] == -1).all()
    aug = _gt_score(d)
    assert np.isclose(aug.hpwl, base.hpwl, atol=1e-2)
    assert np.isclose(aug.area, base.area, atol=1e-2)
    per_group = CondDropoutOp(cluster=0.5, cluster_per_group=True)
    after = []
    for seed in range(30):
        cluster = per_group.apply(inst, np.random.default_rng(seed)).cluster_id
        after.append(len(np.unique(cluster[cluster > 0])))
    before = len(np.unique(inst.cluster_id[inst.cluster_id > 0]))
    assert 0 < np.mean(after) < before


def test_latent_param_bijection():
    """``s_only`` round-trips exactly (numpy and torch)."""
    p = build("s_only", LATENT_PARAM)
    z_np = np.random.randn(5, 3).astype(np.float32)
    assert np.array_equal(p.to_latent(z_np), z_np)
    assert np.allclose(p.from_latent(p.to_latent(z_np)), z_np, atol=1e-6)
    z_t = torch.randn(2, 5, 3)
    assert torch.allclose(p.from_latent(p.to_latent(z_t)), z_t, atol=1e-6)


@needs_data
def test_augment_pipeline_label_preserving():
    inst = load_validation_case(60)
    base = _gt_score(inst)
    pipe = build_pipeline([{"rot90": {"choices": [1, 2, 3]}}, {"flip": {"p": 1.0}}])
    r = _gt_score(pipe.apply(inst, np.random.default_rng(1)))
    assert np.isclose(r.cost, base.cost, atol=1e-3)
    assert np.isclose(r.v_rel, base.v_rel, atol=1e-3)
    assert np.isclose(r.hpwl, base.hpwl, atol=1e-2)


@needs_data
def test_augment_rot_flip_keeps_positions_in_frame():
    """Rotated / flipped cases keep their latent positions non-negative."""
    inst = load_validation_case(60)
    for rot, flip in [(1, False), (2, False), (3, False), (0, True)]:
        t = transform_instance(inst, rot=rot, flip=flip)
        z = xywh_to_z(t.gt_positions, t.area_targets, t.s)
        assert z[:, :2].min() >= -1e-4


def test_augment_pipeline_accepts_all_spec_forms():
    for spec in [
        None,
        False,
        True,
        ["flip"],
        [{"rot90": {"choices": [0, 1, 2, 3]}}],
        [{"name": "shift", "std": 0.05}],
        [{"rot90": {}}, {"flip": {}}, {"shift": {}}],
    ]:
        build_pipeline(spec)
    with pytest.raises(TypeError, match="AUGMENT must be"):
        build_pipeline({"shift_std": 0.1, "rotate": True})


@needs_data
def test_augment_shift_keeps_signals_consistent():
    """A shift moves z0, the anchor and the anchor feature columns together."""
    inst = load_validation_case(60)
    param, pipe = _resolve("s_only", [{"shift": {"std": 0.1}}])
    enc = encode_instance(inst, param, pipe, np.random.default_rng(0))
    z0, az, feats = enc["z0"].numpy(), enc["anchor_z"].numpy(), enc["features"].numpy()
    preplaced = np.nonzero(inst.is_preplaced)[0]
    if preplaced.size:
        i = preplaced[0]
        assert np.allclose(z0[i, :2], az[i, :2], atol=1e-4)
        assert np.allclose(az[i, :2], feats[i, 10:12], atol=1e-4)


@needs_gsrc
def test_gsrc_hard_loads_as_feasible_instance():
    """A GSRC HARD case parses with fixed shapes and an overlap-free reference layout."""
    inst = load_gsrc_case(100, "HARD", allow_download=False)
    assert inst.block_count == 100
    assert inst.area_targets.shape == (100,)
    assert inst.constraints.shape == (100, 5)
    assert inst.b2b.shape[1] == 3 and inst.p2b.shape[1] == 3
    assert bool(inst.is_fixed.all()) and not inst.is_preplaced.any()
    soft = load_gsrc_case(100, "HARD", allow_download=False, hard_shapes=False)
    assert bool(soft.is_soft.all())
    np.testing.assert_allclose(
        inst.area_targets, inst.gt_positions[:, 2] * inst.gt_positions[:, 3], rtol=1e-5
    )
    assert validate(ground_truth_placement(inst), "full").feasible
    assert inst.b2b[:, :2].max() < inst.block_count
    assert inst.p2b[:, 0].max() < inst.pins_pos.shape[0]
