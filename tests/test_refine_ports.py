"""The ported refiners: sizes and aspects untouched, preplaced blocks fixed, padding
untouched, overlap reduced on an overlapping batch; the positions-only closed-form
refiner keeps ``rho``.
"""

import torch

import trinity  # noqa: F401
from tests.test_refine_closed import _case
from trinity.decode import z_to_xywh
from trinity.losses import constraint as C
from trinity.sampling.refine_closed import TERM_NAMES, refine_closed
from trinity_baselines.ports.refine import PORTS

ALL = {k: 1.0 for k in TERM_NAMES}


def _overlap(z, case):
    g = z_to_xywh(z, case.area_norm, z.new_ones(z.shape[0]))
    return C.overlap_area(g, case.token_mask)


def test_ports_move_positions_only_and_reduce_overlap():
    z, case = _case(b=3, n=10, pad=2, seed=11)
    z[..., :2] *= 0.15
    before = _overlap(z, case)
    for name, port in PORTS.items():
        out = port(z, case, 60)
        assert out.shape == z.shape
        assert torch.allclose(out[..., 2], z[..., 2]), name
        assert torch.allclose(out[:, 0, :2], z[:, 0, :2]), name
        assert torch.allclose(out[:, 10:], z[:, 10:]), name
        after = _overlap(out, case)
        assert torch.all(after < before), (name, before, after)


def test_port_snapshots_leave_the_trajectory_unchanged():
    z, case = _case(seed=13, dtype=torch.float32)
    for name, steps in (("chipd_scheduled", 40), ("diffplace", 30), ("macrodiff", 30)):
        plain = PORTS[name](z, case, steps)
        snaps = PORTS[name](z, case, steps, snapshots=(0, 10, steps))
        assert set(snaps) == {0, 10, steps}, name
        assert torch.equal(snaps[steps], plain), name
        assert torch.allclose(snaps[0], z, atol=1e-5) and torch.equal(
            snaps[0][..., 2], z[..., 2]
        ), name
        assert not torch.equal(snaps[10], snaps[steps]), name


def test_positions_only_closed_form_keeps_rho():
    z, case = _case(seed=12)
    out = refine_closed(
        z, case, ALL, steps=10, lr=1e-3, compile_loop=False, shape=False
    )[10]
    clamped = torch.where(case.anchor_mask > 0.5, case.anchor_z, z)
    assert torch.allclose(out[..., 2], clamped[..., 2])
    assert not torch.allclose(out[..., :2], clamped[..., :2])
