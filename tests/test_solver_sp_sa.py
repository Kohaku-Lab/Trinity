"""Gates of the ported sequence-pair SA: decode correctness and an end-to-end solve."""

import numpy as np
import pytest

import trinity_baselines.classical  # noqa: F401  (register: sp_sa)
from trinity.floorplan.data import find_floorset_root, load_validation_case
from trinity.floorplan.geometry import overlapping_pairs
from trinity.floorplan.registry import SOLVER, build
from trinity.floorplan.scoring.cost import hpwl_fast
from trinity.floorplan.types import Placement
from trinity_baselines.classical.sp_sa import _shapes, decode_sequence_pair


def have_data() -> bool:
    try:
        find_floorset_root(allow_download=False)
        return True
    except FileNotFoundError:
        return False


needs_data = pytest.mark.skipif(
    not have_data(), reason="FloorSet validation set not present"
)


def test_decode_two_blocks_horizontal_and_vertical():
    wh = np.array([[2.0, 1.0], [3.0, 1.0]])
    # same order in both sequences -> 1 is right of 0
    xy = decode_sequence_pair(np.array([0, 1]), np.array([0, 1]), wh, (0.0, 0.0))
    assert np.allclose(xy, [[0, 0], [2, 0]])
    # reversed in seq1 only -> 1 is above 0
    xy = decode_sequence_pair(np.array([1, 0]), np.array([0, 1]), wh, (0.0, 0.0))
    assert np.allclose(xy, [[0, 0], [0, 1]])


def test_decode_random_is_overlap_free():
    rng = np.random.default_rng(0)
    n = 30
    wh = rng.uniform(1, 5, size=(n, 2))
    for _ in range(20):
        xy = decode_sequence_pair(
            rng.permutation(n), rng.permutation(n), wh, (0.0, 0.0)
        )
        xywh = np.concatenate([xy, wh], axis=1)
        assert overlapping_pairs(xywh) == []


@needs_data
def test_sp_sa_solves_and_improves():
    inst = load_validation_case(21)
    solver = build({"name": "sp_sa", "max_evals": 300, "seed": 0}, SOLVER)
    out = solver.solve(inst)
    assert out.xywh.shape == (21, 4)
    assert overlapping_pairs(out.xywh) == []
    assert np.allclose(out.xywh[:, 2] * out.xywh[:, 3], inst.area_targets, rtol=1e-4)
    # A 300-evaluation anneal beats a random decode on average.
    rng = np.random.default_rng(1)
    wh = _shapes(inst)
    randoms = []
    for _ in range(10):
        xy = decode_sequence_pair(
            rng.permutation(21), rng.permutation(21), wh, (0.0, 0.0)
        )
        layout = Placement(xywh=np.concatenate([xy, wh], axis=1), instance=inst)
        randoms.append(hpwl_fast(layout))
    assert hpwl_fast(out) < np.mean(randoms)
